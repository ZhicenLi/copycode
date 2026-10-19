"""FedDS stage: independent model, DS reliability, weighted aggregation.

This trainer is deliberately parameterised by two things so that the four
experimental arms share one implementation:

    labels      "noisy" (before correction) or "corrected" (after FedCorr)
    aggregate   "fedavg" (data-size weights) or "fedds" (DS reliability)

That is what makes the 2x2 comparison table possible without duplicated code:
the FedCorr stage only ever produces a label file, and this stage never sees a
FedCorr model.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..core.dawid_skene import dawid_skene
from ..data.loaders import build_transform, evaluate, make_loader, predict_labels
from ..models.registry import build_model, describe
from ..seeding import rng as make_rng


@dataclass
class FedDSResult:
    rounds: List[Dict] = field(default_factory=list)
    final_accuracy: float = 0.0
    last_k_accuracy: float = 0.0
    client_reliability: Dict[int, List[float]] = field(default_factory=dict)


def weighted_average(states: List[Dict[str, torch.Tensor]], weights: np.ndarray):
    """Weighted average of state dicts, handling integer buffers explicitly."""
    if len(states) != len(weights):
        raise ValueError(
            f"{len(states)} client states but {len(weights)} weights were supplied"
        )
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.ones(len(states), dtype=np.float64) / len(states)
    else:
        weights = np.asarray(weights, dtype=np.float64) / total

    averaged: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        reference = states[0][key]
        # the weight tensor must live on the same device as the parameters,
        # otherwise the first aggregation on CUDA raises a device mismatch
        shape = (-1,) + (1,) * reference.dim()
        weight = torch.as_tensor(
            weights, dtype=torch.float64, device=reference.device
        ).view(*shape)
        stacked = torch.stack([state[key].double() for state in states])
        value = (stacked * weight).sum(0)
        if torch.is_floating_point(reference):
            averaged[key] = value.to(reference.dtype)
        else:
            # BatchNorm's num_batches_tracked is int64: average, then round back
            averaged[key] = value.round().to(reference.dtype)
    return averaged


class FedDSTrainer:
    def __init__(
        self,
        cfg: Config,
        manifest,
        train_images: torch.Tensor,
        test_loader,
        public_loader,
        labels: np.ndarray,
        device: torch.device,
        output_dir: str,
        aggregate: str = "fedds",
        tag: str = "fedds",
    ):
        if aggregate not in ("fedavg", "fedds"):
            raise ValueError(f"unknown aggregation: {aggregate}")
        self.cfg = cfg
        self.manifest = manifest
        self.images = train_images
        self.test_loader = test_loader
        self.public_loader = public_loader
        self.labels = np.asarray(labels)
        self.device = device
        self.output_dir = output_dir
        self.aggregate = aggregate
        self.tag = tag
        os.makedirs(output_dir, exist_ok=True)

        self.spec = cfg.data.spec()
        self.num_clients = cfg.data.num_clients
        self.model = build_model(
            cfg.model.fedds, cfg.data.num_classes, self.spec.channels,
            pretrained=cfg.model.fedds_pretrained,
        ).to(device)
        self.local = build_model(
            cfg.model.fedds, cfg.data.num_classes, self.spec.channels,
            pretrained=cfg.model.fedds_pretrained,
        ).to(device)
        # recorded so a paper table can prove every arm started from the same weights
        self.initial_state_sha = _state_digest(self.model.state_dict())

        self._train_tf = build_transform(self.spec, "train", cfg.fedds.preprocess)
        self.loss_fn = nn.CrossEntropyLoss()

        self.client_indices = {
            client: manifest.client_indices(client) for client in range(self.num_clients)
        }
        self.reliability_history: Dict[int, List[float]] = {}

    # ---------------------------------------------------------------- helpers
    def _local_train(self, indices: np.ndarray, seed: int, lr: float) -> float:
        cfg = self.cfg.fedds
        # A client update must be a pure function of (round, client). Dropout and
        # the torchvision crop/flip draw from the *global* RNGs, so without this a
        # resumed run would replay a different stream than an uninterrupted one.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        loader = make_loader(
            self.images, self.labels, indices,
            batch_size=cfg.local_bs, train=True, transform=self._train_tf,
            num_workers=self.cfg.runtime.num_workers, seed=seed,
        )
        optimizer = torch.optim.SGD(
            self.local.parameters(), lr=lr, momentum=cfg.momentum
        )
        self.local.train()
        losses: List[float] = []
        for _ in range(cfg.local_ep):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                optimizer.zero_grad()
                loss = self.loss_fn(self.local(images), labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _lr_for_round(self, round_index: int) -> float:
        """Cosine decay when enabled, constant learning rate otherwise.

        The authoritative FedDS run uses a constant lr, so `lr_decay` stays
        False by default and the main results remain faithful.
        """
        base = self.cfg.fedds.lr
        if not self.cfg.fedds.lr_decay:
            return base
        span = max(self.cfg.fedds.rounds - 1, 1)
        progress = (round_index - 1) / span
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _public_predictions(self, model: torch.nn.Module) -> np.ndarray:
        """Hard predictions on the cached public tensor (deterministic transform)."""
        return predict_labels(model, self.public_loader, self.device)

    def _run_fingerprint(self) -> str:
        """Identify the exact training setup a checkpoint belongs to.

        `rounds` is deliberately excluded: extending a run from 100 to 200 rounds
        is the main reason to resume, and must not invalidate the checkpoint.
        """
        import dataclasses
        import hashlib

        fedds = dataclasses.asdict(self.cfg.fedds)
        # `rounds` normally stays out so a run can be extended; but with lr decay
        # the schedule depends on it, so then it must invalidate the checkpoint
        if not self.cfg.fedds.lr_decay:
            fedds.pop("rounds", None)
        payload = {
            "fedds": fedds,
            "model": self.cfg.model.fedds,
            "pretrained": self.cfg.model.fedds_pretrained,
            "num_classes": self.cfg.data.num_classes,
            "preprocess": self.cfg.fedds.preprocess,
            "seed": self.cfg.runtime.seed,
            "num_workers": self.cfg.runtime.num_workers,
            "deterministic": self.cfg.runtime.deterministic,
            "manifest": self.manifest.digest(),
            "labels": hashlib.sha256(
                np.ascontiguousarray(self.labels).tobytes()
            ).hexdigest(),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # -------------------------------------------------------------------- run
    def run(self) -> FedDSResult:
        cfg = self.cfg.fedds
        rounds: List[Dict] = []
        start = time.time()
        per_round = cfg.clients_per_round(self.num_clients)

        log_path = os.path.join(self.output_dir, f"{self.tag}_rounds.jsonl")
        resume_path = os.path.join(self.output_dir, f"{self.tag}_resume.pt")

        start_round = 1
        resumed = False
        fingerprint = self._run_fingerprint()
        if self.cfg.runtime.resume and os.path.exists(resume_path):
            payload = torch.load(resume_path, map_location=self.device, weights_only=False)
            if payload.get("fingerprint") != fingerprint:
                print(f"[resume] {self.tag}: hyper-parameters changed, starting fresh")
                os.remove(resume_path)          # never leave a checkpoint we refused
            elif not np.array_equal(np.asarray(payload["labels"]), self.labels):
                print(f"[resume] {self.tag}: checkpoint labels differ, starting fresh")
                os.remove(resume_path)
            else:
                self.model.load_state_dict(payload["model"])
                start_round = int(payload["round"]) + 1
                resumed = True
                print(f"[resume] {self.tag}: continuing from round {start_round}")

        if start_round > cfg.rounds:
            raise ValueError(
                f"checkpoint is at round {start_round - 1} but fedds.rounds={cfg.rounds}; "
                "raise fedds.rounds or set runtime.resume=false"
            )

        # a fresh run must not append to a previous log; a resumed one must keep it
        if not resumed and os.path.exists(log_path):
            os.remove(log_path)
        elif resumed and os.path.exists(log_path):
            # the log may contain a round that was written before the crash but
            # never checkpointed; drop anything past the checkpoint
            kept = []
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        if int(json.loads(line).get("round", 0)) < start_round:
                            kept.append(line)
                    except (ValueError, AttributeError):
                        continue
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.writelines(kept)

        for round_index in range(start_round, cfg.rounds + 1):
            round_rng = make_rng(self.cfg.runtime.seed, "fedds_round", round_index)
            selected = round_rng.choice(self.num_clients, per_round, replace=False)
            learning_rate = self._lr_for_round(round_index)

            # every selected client must start from the *same* round-global model;
            # keeping an explicit snapshot makes that invariant impossible to break
            global_state = {
                key: value.detach().clone() for key, value in self.model.state_dict().items()
            }

            states: List[Dict[str, torch.Tensor]] = []
            local_losses: List[float] = []
            data_sizes: List[int] = []
            predictions: List[np.ndarray] = []

            for client in selected:
                client = int(client)
                indices = self.client_indices[client]
                self.local.load_state_dict(global_state)
                seed = int(
                    make_rng(self.cfg.runtime.seed, "fedds_local", round_index, client)
                    .integers(0, 2 ** 31 - 1)
                )
                local_losses.append(self._local_train(indices, seed, learning_rate))
                states.append({k: v.detach().clone() for k, v in self.local.state_dict().items()})
                data_sizes.append(int(len(indices)))

                # predict with the just-trained local model; the global model stays
                # untouched until aggregation (otherwise clients would chain)
                predictions.append(self._public_predictions(self.local))

            record: Dict[str, object] = {
                "round": round_index,
                "arm": self.tag,
                "selected_clients": [int(c) for c in selected],
                "local_loss_mean": float(np.mean(local_losses)),
            }

            if self.aggregate == "fedds":
                matrix = np.stack(predictions, axis=1)          # (items, annotators)
                ds = dawid_skene(
                    matrix,
                    num_classes=self.cfg.data.num_classes,
                    max_iter=cfg.ds_max_iter,
                    tol=cfg.ds_tol,
                    use_class_prior=cfg.ds_use_class_prior,
                )
                weights = ds.reliability.astype(np.float64)
                record.update({
                    "reliability": [float(w) for w in weights],
                    "ds_iterations": ds.iterations,
                    "ds_converged": bool(ds.converged),
                    "ds_zero_support_rows": int(ds.zero_support_rows),
                    "annotators": int(matrix.shape[1]),
                })
                for client, weight in zip(selected, weights):
                    self.reliability_history.setdefault(int(client), []).append(float(weight))
            else:
                weights = np.asarray(data_sizes, dtype=np.float64)
                record.update({"annotators": 0})

            averaged = weighted_average(states, weights)
            self.model.load_state_dict(averaged)

            accuracy = evaluate(self.model, self.test_loader, self.device)
            record.update({
                "test_accuracy": accuracy,
                "elapsed_s": round(time.time() - start, 1),
                "normalized_weight_max": float(np.max(weights) / max(np.sum(weights), 1e-12)),
            })
            rounds.append(record)
            _append_jsonl(os.path.join(self.output_dir, f"{self.tag}_rounds.jsonl"), record)

            if self.cfg.runtime.resume:
                tmp_path = resume_path + ".tmp"
                torch.save({
                    "round": round_index,
                    "model": self.model.state_dict(),
                    "labels": self.labels,
                    "fingerprint": fingerprint,
                }, tmp_path)
                os.replace(tmp_path, resume_path)

            if self.cfg.runtime.save_every and round_index % self.cfg.runtime.save_every == 0:
                self._checkpoint(round_index)

        # read the whole log so a resumed run still reports the full curve
        accuracies: List[float] = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        accuracies.append(float(json.loads(line)["test_accuracy"]))
                    except (KeyError, ValueError):
                        continue
        # a checkpoint could hold more rounds than the current config asks for
        accuracies = accuracies[-cfg.rounds:]

        last_k = accuracies[-cfg.last_k:] if accuracies else []
        summary = FedDSResult(
            rounds=rounds,
            final_accuracy=float(accuracies[-1]) if accuracies else 0.0,
            last_k_accuracy=float(np.mean(last_k)) if last_k else 0.0,
            client_reliability=self.reliability_history,
        )
        with open(os.path.join(self.output_dir, f"{self.tag}_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "arm": self.tag,
                "rounds": cfg.rounds,
                "resumed": resumed,
                "initial_state_sha": self.initial_state_sha,
                "final_accuracy": summary.final_accuracy,
                f"last_{cfg.last_k}_accuracy": summary.last_k_accuracy,
                "accuracy_curve": accuracies,
                "model": describe(self.model),
            }, fh, indent=2, ensure_ascii=False)
        return summary

    def _checkpoint(self, round_index: int) -> None:
        if not self.cfg.runtime.checkpoint_models:
            return
        torch.save(
            {"round": round_index, "model": self.model.state_dict(), "labels": self.labels},
            os.path.join(self.output_dir, f"{self.tag}_ckpt_r{round_index}.pt"),
        )


def _state_digest(state: Dict[str, torch.Tensor]) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode("utf-8"))
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def _append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
