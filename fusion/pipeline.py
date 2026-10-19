"""Stage orchestration: data -> FedCorr -> label snapshot -> FedDS arms.

The two stages never share a model. The only thing crossing the boundary is a
label array keyed by `sample_id`, saved as a separate snapshot so the manifest
itself stays immutable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .analysis import format_table, summarise
from .config import Config
from .data.loaders import (
    build_transform,
    load_cifar,
    make_cached_loader,
    make_loader,
    synthetic_source,
)
from .data.manifest import Manifest, build_manifest, noise_report
from .seeding import derive_seed
from .stages.fedcorr_preprocess import FedCorrPreprocessor
from .stages.fedds_train import FedDSTrainer

ARM_SPEC = {
    "fedavg": {"labels": "noisy", "aggregate": "fedavg"},
    "fedds": {"labels": "noisy", "aggregate": "fedds"},
    "fedcorr+fedavg": {"labels": "corrected", "aggregate": "fedavg"},
    "fedcorr+fedds": {"labels": "corrected", "aggregate": "fedds"},
}


def prepare_data(cfg: Config, run_dir: str, synthetic: bool = False):
    spec = cfg.data.spec()
    if synthetic:
        images, labels = synthetic_source(240, spec.num_classes, spec.image_size, seed=0)
        test_images, test_labels = synthetic_source(96, spec.num_classes, spec.image_size, seed=1)
    else:
        images, labels = load_cifar(
            cfg.data.dataset, cfg.data.data_root, train=True, download=cfg.data.download
        )
        test_images, test_labels = load_cifar(
            cfg.data.dataset, cfg.data.data_root, train=False, download=cfg.data.download
        )

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "manifest.npz")

    if os.path.exists(manifest_path):
        manifest = Manifest.load(data_dir)
        _check_manifest_matches(manifest, cfg)
    else:
        manifest = build_manifest(
            clean_labels=labels,
            spec=spec,
            num_clients=cfg.data.num_clients,
            dirichlet_alpha=cfg.data.dirichlet_alpha,
            public_ratio=cfg.data.public_ratio,
            noise_levels=cfg.data.noise_levels,
            split_seed=cfg.data.split_seed,
            noise_seed=cfg.data.noise_seed,
            min_client_samples=cfg.data.min_client_samples,
            dirichlet_max_tries=cfg.data.dirichlet_max_tries,
            partition=cfg.data.partition,
            noise_rates=cfg.data.noise_rates,
            public_split=cfg.data.public_split,
        )
        manifest.save(data_dir)
        with open(os.path.join(data_dir, "noise_report.json"), "w", encoding="utf-8") as fh:
            json.dump(noise_report(manifest), fh, indent=2)

    if len(manifest.public_indices()) == 0:
        raise ValueError(
            "the public split is empty; FedDS cannot estimate reliability. "
            "Raise data.public_ratio or use a larger dataset."
        )
    if len(test_labels) == 0:
        raise ValueError("the evaluation split is empty; cannot measure accuracy")

    # Each stage keeps its own paper's data pipeline: FedCorr trains and scores
    # with crop/flip/normalise, FedDS uses raw x/255. The evaluation loader must
    # match the preprocessing its model was trained with, so build one per stage.
    test_loaders = {}
    for mode in sorted({cfg.fedcorr.preprocess, cfg.fedds.preprocess}):
        transform = build_transform(spec, "eval", mode)
        if cfg.runtime.cache_eval:
            loader, _ = make_cached_loader(
                test_images, test_labels, np.arange(len(test_labels)),
                transform, batch_size=256,
            )
        else:
            loader = make_loader(
                test_images, test_labels, np.arange(len(test_labels)),
                batch_size=256, train=False, transform=transform,
            )
        test_loaders[mode] = loader

    public_transform = build_transform(spec, "eval", cfg.fedds.preprocess)
    # the public split carries no labels by construction; zeros are placeholders
    if cfg.runtime.cache_eval:
        public_loader, _ = make_cached_loader(
            images, np.zeros(len(images), dtype=np.int64), manifest.public_indices(),
            public_transform, batch_size=cfg.fedds.public_bs,
        )
    else:
        public_loader = make_loader(
            images, np.zeros(len(images), dtype=np.int64), manifest.public_indices(),
            batch_size=cfg.fedds.public_bs, train=False, transform=public_transform,
        )
    return images, labels, manifest, test_loaders, public_loader


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training path touches, so a run is reproducible.

    Model initialisation and dropout use the global torch RNG; without this a
    rerun with the same config produced different weights.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(cfg: Config, synthetic: Optional[bool] = None, device: Optional[str] = None) -> Dict:
    if synthetic is None:
        synthetic = cfg.runtime.synthetic
    run_dir = cfg.run_dir()
    os.makedirs(run_dir, exist_ok=True)
    experiment_start = time.time()
    cfg.save(os.path.join(run_dir, "config.json"))

    resolved_device = torch.device(device or cfg.runtime.device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        if not cfg.runtime.allow_cpu_fallback:
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "A 200-round run on CPU would take days, so this is refused. "
                "Fix the CUDA install, or set runtime.allow_cpu_fallback=true "
                "if you really intend to run on CPU."
            )
        print("[warn] CUDA requested but unavailable; falling back to CPU "
              "(runtime.allow_cpu_fallback=true)")
        resolved_device = torch.device("cpu")

    images, labels, manifest, test_loaders, public_loader = prepare_data(
        cfg, run_dir, synthetic=synthetic
    )
    _verify_manifest_labels(manifest, labels)
    print(f"data ready: {len(labels)} train / public {len(manifest.public_indices())} / "
          f"{cfg.data.num_clients} clients")

    corrected: Optional[np.ndarray] = None
    fedcorr_stats: Optional[Dict] = None
    fedcorr_dir = os.path.join(run_dir, "fedcorr")
    needs_correction = any(
        ARM_SPEC[arm]["labels"] == "corrected" for arm in cfg.runtime.arms
    )
    if needs_correction:
        fingerprint = _correction_fingerprint(cfg, manifest)
        labels_path = os.path.join(fedcorr_dir, "corrected_labels.npy")
        meta_path = os.path.join(fedcorr_dir, "correction_meta.json")
        analysis_path = os.path.join(fedcorr_dir, "analysis.json")

        reuse = False
        if cfg.runtime.reuse_correction and os.path.exists(labels_path) and os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("fingerprint") == fingerprint:
                corrected = np.load(labels_path)
                reuse = True
                print(f"reusing corrected labels ({saved.get('iterations', '?')} iterations)")

        analysis: Optional[Dict] = None
        if reuse:
            if os.path.exists(analysis_path):
                with open(analysis_path, encoding="utf-8") as fh:
                    saved_analysis = json.load(fh)
                if saved_analysis.get("fingerprint") == fingerprint:
                    analysis = saved_analysis
                    print(format_table(analysis))
                else:
                    print(
                        "[warn] analysis.json belongs to a different FedCorr run; "
                        "diagnostics are unavailable for the reused labels"
                    )
        else:
            seed_everything(
                derive_seed(cfg.runtime.seed, "stage", "fedcorr"),
                deterministic=cfg.runtime.deterministic,
            )
            preprocessor = FedCorrPreprocessor(
                cfg, manifest, images,
                test_loaders[cfg.fedcorr.preprocess],
                resolved_device, fedcorr_dir,
            )
            result = preprocessor.run()
            corrected = result.corrected_labels
            np.save(labels_path, corrected)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"fingerprint": fingerprint,
                           "iterations": cfg.fedcorr.iteration1}, fh, indent=2)

            # ground-truth diagnostics (clean labels are used here only, never in training)
            analysis = summarise(manifest, corrected, result.per_iteration)
            analysis["fingerprint"] = fingerprint
            with open(analysis_path, "w", encoding="utf-8") as fh:
                json.dump(analysis, fh, indent=2, ensure_ascii=False)
            print(format_table(analysis))

        n_corrected = int(np.sum(corrected != manifest.noisy_label))
        fedcorr_stats = {
            "reused": reuse,
            "corrected_samples": n_corrected,
            "corrected_ratio": n_corrected / max(len(corrected), 1),
            "iterations": cfg.fedcorr.iteration1,
            "analysis": analysis,
        }
        print(f"FedCorr done: {n_corrected} labels corrected "
              f"({100.0 * fedcorr_stats['corrected_ratio']:.1f}%)")

        if n_corrected == 0 and not synthetic:
            print(
                "[warn] FedCorr corrected nothing. The correction stage had no effect, "
                "so 'fedcorr+*' arms will be identical to their noisy-label counterparts. "
                "Check confidence_thres / iteration1 / local_ep before trusting the result."
            )
        if analysis and analysis.get("correction") and not synthetic:
            precision = analysis["correction"].get("correction_precision")
            if precision is not None and precision == precision and precision < 0.5:
                print(
                    f"[warn] correction precision is only {precision:.2f}: the model is "
                    "breaking more labels than it fixes. Treat this run as a smoke test, "
                    "not as a result."
                )

    table: List[Dict] = []
    for arm in cfg.runtime.arms:
        spec = ARM_SPEC[arm]
        arm_labels = corrected if spec["labels"] == "corrected" else manifest.noisy_label
        if arm_labels is None:
            raise RuntimeError(f"arm '{arm}' needs corrected labels but FedCorr did not run")

        tag = arm.replace("+", "_")
        # All arms share one seed: same initial weights, same client schedule and
        # same dropout stream, so a difference between arms comes from the method
        # (labels / aggregation), not from initialisation noise.
        seed_everything(
            derive_seed(cfg.runtime.seed, "stage", "fedds"),
            deterministic=cfg.runtime.deterministic,
        )
        trainer = FedDSTrainer(
            cfg=cfg,
            manifest=manifest,
            train_images=images,
            test_loader=test_loaders[cfg.fedds.preprocess],
            public_loader=public_loader,
            labels=arm_labels.copy(),
            device=resolved_device,
            output_dir=os.path.join(run_dir, "runs", tag),
            aggregate=spec["aggregate"],
            tag=tag,
        )
        summary = trainer.run()
        row = {
            "arm": arm,
            "labels": spec["labels"],
            "aggregate": spec["aggregate"],
            "final_accuracy": summary.final_accuracy,
            f"last{cfg.fedds.last_k}_accuracy": summary.last_k_accuracy,
            "annotators_per_round": (
                cfg.annotators_per_round() if spec["aggregate"] == "fedds" else 0
            ),
        }
        table.append(row)
        print(f"  {arm:18s} final={summary.final_accuracy:.2f}  "
              f"last{cfg.fedds.last_k}={summary.last_k_accuracy:.2f}")

    payload = {
        "experiment": cfg.runtime.exp_name,
        "dataset": cfg.data.dataset,
        "num_clients": cfg.data.num_clients,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "public_ratio": cfg.data.public_ratio,
        "public_samples": int(len(manifest.public_indices())),
        "noise_levels": list(cfg.data.noise_levels),
        "local_ep": cfg.fedds.local_ep,
        "rounds": cfg.fedds.rounds,
        "annotators_per_round": cfg.annotators_per_round(),
        "fedcorr": fedcorr_stats,
        "elapsed_minutes": round((time.time() - experiment_start) / 60.0, 2),
        "results": table,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(run_dir, "summary.csv"), table)
    print(f"summary -> {os.path.join(run_dir, 'summary.json')}")
    return payload


def _correction_fingerprint(cfg: Config, manifest: Manifest) -> str:
    """Identify a correction result: same config + same data => safe to reuse."""
    import dataclasses
    import hashlib

    payload = {
        "fedcorr": dataclasses.asdict(cfg.fedcorr),
        "model": cfg.model.fedcorr,
        "dataset": cfg.data.dataset,
        "num_clients": cfg.data.num_clients,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "public_ratio": cfg.data.public_ratio,
        "split_seed": cfg.data.split_seed,
        "noise_seed": cfg.data.noise_seed,
        "preprocess": cfg.fedcorr.preprocess,
        "seed": cfg.runtime.seed,          # client order and local seeds depend on it
        "num_workers": cfg.runtime.num_workers,
        "deterministic": cfg.runtime.deterministic,
        "manifest": manifest.digest(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_manifest_labels(manifest: Manifest, clean_labels: np.ndarray) -> None:
    """Guard against pointing a cached manifest at a different dataset."""
    if len(manifest.clean_label) != len(clean_labels):
        raise ValueError(
            f"cached manifest has {len(manifest.clean_label)} samples but the dataset "
            f"has {len(clean_labels)}; the data root or experiment name changed"
        )
    if not np.array_equal(manifest.clean_label, clean_labels):
        raise ValueError(
            "cached manifest does not match the dataset labels at this data root; "
            "refusing to train on a stale partition"
        )


def _check_manifest_matches(manifest: Manifest, cfg: Config) -> None:
    expected = {
        "dataset": cfg.data.dataset,
        "num_classes": cfg.data.num_classes,
        "num_clients": cfg.data.num_clients,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "public_ratio": cfg.data.public_ratio,
        "split_seed": cfg.data.split_seed,
        "noise_seed": cfg.data.noise_seed,
        "min_client_samples": cfg.data.min_client_samples,
        "noise_levels": [float(v) for v in cfg.data.noise_levels],
        "partition": cfg.data.partition,
        "public_split": cfg.data.public_split,
    }
    if cfg.data.noise_rates is not None:
        expected["client_noise_rates"] = [float(v) for v in cfg.data.noise_rates]
    for key, value in expected.items():
        actual = manifest.meta.get(key)
        if actual is None:
            raise ValueError(
                f"cached manifest has no '{key}' record; it predates this config or was "
                "written by a different version. Delete the output directory and rerun."
            )
        if actual != value:
            raise ValueError(
                f"cached manifest has {key}={actual} but the config asks for {value}; "
                "use a different experiment name or delete the cached manifest"
            )


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(str(row[key]) for key in keys) + "\n")
