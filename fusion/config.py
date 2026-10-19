"""Typed configuration for the FedCorr -> FedDS pipeline.

Every hyper-parameter the study sweeps lives here and can be overridden from
the command line, so no experiment value is hard-coded in the algorithm files.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data.specs import get_spec


DEFAULT_NOISE_LEVELS: Tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

DEFAULT_ARMS: Tuple[str, ...] = (
    "fedavg",
    "fedds",
    "fedcorr+fedavg",
    "fedcorr+fedds",
)


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_root: str = "data"
    num_classes: int = 0                      # 0 -> resolved from the dataset spec
    num_clients: int = 100
    dirichlet_alpha: float = 0.5
    partition: str = "dirichlet"    # "dirichlet" | "iid" (iid = random equal split, official FedDS)
    public_ratio: float = 0.1
    noise_levels: Tuple[float, ...] = DEFAULT_NOISE_LEVELS
    noise_rates: Optional[Tuple[float, ...]] = None   # explicit per-client rates (official FedDS)
    public_split: str = "stratified"   # "stratified" (ours) | "sklearn" (official train_test_split)
    split_seed: int = 42
    noise_seed: int = 2026
    download: bool = True          # False when the data is already staged on the box
    min_client_samples: int = 10
    dirichlet_max_tries: int = 1000

    def spec(self):
        return get_spec(self.dataset)


@dataclass
class ModelConfig:
    fedcorr: str = ""          # "" -> resolved from the dataset spec
    fedds: str = ""            # "" -> resolved from the dataset spec
    fedds_pretrained: bool = False   # keep False so both stages start from scratch


@dataclass
class FedCorrConfig:
    """Port of FedCorr's federated pre-processing stage."""

    iteration1: int = 5
    frac1: Optional[float] = None   # None -> 1 / num_clients
    local_ep: int = 5
    local_bs: int = 100
    lr: float = 0.03
    momentum: float = 0.5
    beta: float = 5.0               # proximal coefficient
    mixup: bool = True
    mixup_alpha: float = 1.0
    lid_k: int = 20
    relabel_ratio: float = 0.5
    confidence_thres: float = 0.5
    correct: bool = True            # run the label-correction pass each iteration
    train_transform: str = "train"  # "train" | "eval"; FedCorr scores with random crops
    loss_on: str = "logit"          # "logit" = correct CE, "prob" = upstream double-softmax
    preprocess: str = "standard"    # this stage's own data pipeline (FedCorr: crop/flip/normalise)

    def resolved_frac1(self, num_clients: int) -> float:
        if self.frac1 is not None:
            return self.frac1
        return 1.0 / num_clients


@dataclass
class FedDSConfig:
    rounds: int = 200
    frac: float = 0.1               # paper: 10 of 100 clients per round
    frac_by_num_clients: Dict[int, float] = field(default_factory=lambda: {10: 0.3})
    local_ep: int = 5
    local_bs: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    public_bs: int = 64
    last_k: int = 10
    ds_max_iter: int = 500
    ds_tol: float = 1e-5
    ds_use_class_prior: bool = False   # False keeps the authoritative DS exactly
    lr_decay: bool = False
    preprocess: str = "standard"    # this stage's own data pipeline (FedDS paper: raw x/255)

    def clients_per_round(self, num_clients: int) -> int:
        frac = self.frac_by_num_clients.get(num_clients, self.frac)
        return max(int(frac * num_clients), 1)


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    num_workers: int = 0        # raise to 2-4 on the GPU server; 0 avoids worker pipes
    output_root: str = "outputs"
    exp_name: str = "run"
    seed: int = 1
    save_every: int = 20
    arms: Tuple[str, ...] = DEFAULT_ARMS
    synthetic: bool = False     # CPU smoke: replace the dataset with random images
    checkpoint_models: bool = True   # saves full weights; ~45 MB per checkpoint
    cache_eval: bool = True     # pre-transform test/public tensors once (big speedup)
    allow_cpu_fallback: bool = False  # refuse to silently train on CPU when CUDA is asked
    deterministic: bool = True  # cudnn deterministic mode: slower, but reproducible
    reuse_correction: bool = True   # reuse an existing corrected_labels.npy if the fingerprint matches
    resume: bool = False        # continue FedDS from the last round checkpoint


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fedcorr: FedCorrConfig = field(default_factory=FedCorrConfig)
    fedds: FedDSConfig = field(default_factory=FedDSConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_file(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        raw = _read_structured(path)
        if overrides:
            raw = _deep_merge(raw, overrides)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        known = {"data", "model", "fedcorr", "fedds", "runtime"}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(
                f"unknown config section(s): {sorted(unknown)}; expected {sorted(known)}"
            )
        cfg = cls()
        for section in ("data", "model", "fedcorr", "fedds", "runtime"):
            block = raw.get(section) or {}
            target = getattr(cfg, section)
            for key, value in block.items():
                if not hasattr(target, key):
                    raise KeyError(f"unknown config key: {section}.{key}")
                if key == "noise_levels":
                    value = tuple(float(v) for v in _as_list(value))
                elif key == "noise_rates":
                    value = (
                        None
                        if value is None
                        else tuple(float(v) for v in _as_list(value))
                    )
                elif key == "arms":
                    value = tuple(str(v) for v in _as_list(value))
                elif key == "frac_by_num_clients":
                    value = _parse_frac_map(value)
                elif key == "exp_name":
                    value = str(value)
                setattr(target, key, value)
        cfg.resolve_defaults()
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ setup
    def resolve_defaults(self) -> None:
        """Fill fields that depend on the dataset."""
        spec = self.data.spec()
        if self.data.num_classes in (0, None):
            self.data.num_classes = spec.num_classes
        if not self.model.fedcorr:
            self.model.fedcorr = spec.fedcorr_model
        if not self.model.fedds:
            self.model.fedds = spec.fedds_model

    def validate(self) -> None:
        d, f, s = self.data, self.fedcorr, self.fedds
        spec = d.spec()
        if not spec.implemented:
            raise NotImplementedError(
                f"dataset '{d.dataset}' is registered but its loader is not implemented yet"
            )
        _require_int(d.num_clients, "data.num_clients", minimum=2)
        _require_int(d.num_classes, "data.num_classes", minimum=1)
        if d.num_classes != spec.num_classes:
            raise ValueError(
                f"data.num_classes={d.num_classes} contradicts dataset '{d.dataset}' "
                f"({spec.num_classes} classes)"
            )
        if not (0.0 < d.public_ratio < 1.0):
            raise ValueError("data.public_ratio must be in (0, 1)")
        if d.num_clients < 2:
            # the LID grouping is a two-component GMM over clients
            raise ValueError("data.num_clients must be at least 2 (the GMM needs two clients)")
        if not _finite_positive(d.dirichlet_alpha):
            raise ValueError("data.dirichlet_alpha must be a finite positive number")
        if d.partition not in ("dirichlet", "iid"):
            raise ValueError("data.partition must be 'dirichlet' or 'iid'")
        for stage, mode in (("fedcorr", f.preprocess), ("fedds", s.preprocess)):
            if mode not in ("standard", "raw"):
                raise ValueError(
                    f"{stage}.preprocess must be 'standard' (crop/flip/normalise) or "
                    f"'raw' (the original {stage} pipeline: x/255 only)"
                )
        _require_int(d.min_client_samples, "data.min_client_samples", minimum=1)
        _require_int(d.dirichlet_max_tries, "data.dirichlet_max_tries", minimum=1)
        _require_seed(d.split_seed, "data.split_seed")
        _require_seed(d.noise_seed, "data.noise_seed")
        if not d.noise_levels:
            raise ValueError("data.noise_levels must not be empty")
        if any(not (0.0 <= v <= 1.0) for v in d.noise_levels):
            raise ValueError("data.noise_levels must be within [0, 1]")
        if d.noise_rates is not None:
            if len(d.noise_rates) != d.num_clients:
                raise ValueError(
                    f"data.noise_rates has {len(d.noise_rates)} entries but "
                    f"num_clients is {d.num_clients}"
                )
            if any(not (0.0 <= float(v) <= 1.0) for v in d.noise_rates):
                raise ValueError("data.noise_rates must be within [0, 1]")
        if d.public_split not in ("stratified", "sklearn"):
            raise ValueError("data.public_split must be 'stratified' or 'sklearn'")

        if f.frac1 is not None and not _finite_positive(f.frac1):
            raise ValueError("fedcorr.frac1 must be a finite positive number or null")
        frac1 = f.resolved_frac1(d.num_clients)
        picked = int(round(d.num_clients * frac1))
        steps = int(round(1.0 / frac1))
        if abs(d.num_clients * frac1 - picked) > 1e-6:
            raise ValueError(
                f"frac1={frac1} does not select a whole number of clients out of "
                f"{d.num_clients}; use frac1 = 1/num_clients."
            )
        if picked != 1:
            raise ValueError(
                "FedCorr's preprocessing aggregates one client at a time; "
                f"frac1={frac1} selects {picked} clients out of {d.num_clients}."
            )
        if steps != d.num_clients:
            raise ValueError(
                f"int(1/frac1)={steps} must equal num_clients={d.num_clients}, "
                "otherwise one iteration does not visit every client exactly once."
            )
        if f.loss_on not in ("logit", "prob"):
            raise ValueError("fedcorr.loss_on must be 'logit' or 'prob'")
        if f.train_transform not in ("train", "eval"):
            raise ValueError("fedcorr.train_transform must be 'train' or 'eval'")
        _require_int(f.iteration1, "fedcorr.iteration1", minimum=1)
        _require_int(f.local_ep, "fedcorr.local_ep", minimum=1)
        _require_int(f.local_bs, "fedcorr.local_bs", minimum=1)
        _require_int(f.lid_k, "fedcorr.lid_k", minimum=2)
        if not _finite_positive(f.lr):
            raise ValueError("fedcorr.lr must be finite and positive")
        if not _finite_positive(f.mixup_alpha):
            raise ValueError("fedcorr.mixup_alpha must be finite and positive")
        if not (0.0 < f.relabel_ratio <= 1.0):
            raise ValueError("fedcorr.relabel_ratio must be in (0, 1]")
        if not (0.0 <= f.confidence_thres <= 1.0):
            raise ValueError("fedcorr.confidence_thres must be in [0, 1]")

        _require_int(s.rounds, "fedds.rounds", minimum=1)
        _require_int(s.last_k, "fedds.last_k", minimum=1)
        if s.last_k > s.rounds:
            raise ValueError("fedds.last_k cannot exceed fedds.rounds")
        if not (0.0 < s.frac <= 1.0):
            raise ValueError("fedds.frac must be in (0, 1]")
        _require_int(s.local_ep, "fedds.local_ep", minimum=1)
        _require_int(s.local_bs, "fedds.local_bs", minimum=1)
        _require_int(s.public_bs, "fedds.public_bs", minimum=1)
        _require_int(s.ds_max_iter, "fedds.ds_max_iter", minimum=1)
        if not _finite_positive(s.ds_tol):
            raise ValueError("fedds.ds_tol must be finite and positive")
        if not _finite_positive(s.lr):
            raise ValueError("fedds.lr must be finite and positive")
        for client_count, value in s.frac_by_num_clients.items():
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(
                    f"fedds.frac_by_num_clients[{client_count}]={value} must be in (0, 1]"
                )
        per_round = s.clients_per_round(d.num_clients)
        if per_round < 1:
            raise ValueError("fedds would select zero clients per round")
        if per_round > d.num_clients:
            raise ValueError(
                f"fedds selects {per_round} clients per round but only "
                f"{d.num_clients} exist; lower fedds.frac or frac_by_num_clients"
            )
        if not self.runtime.arms:
            raise ValueError("runtime.arms is empty: nothing to train")
        if len(set(self.runtime.arms)) != len(self.runtime.arms):
            raise ValueError("runtime.arms contains duplicates")
        for arm in self.runtime.arms:
            if arm not in DEFAULT_ARMS:
                raise ValueError(f"unknown experiment arm: {arm}")
        if any(arm.endswith("+fedds") or arm == "fedds" for arm in self.runtime.arms):
            if per_round < 2:
                raise ValueError(
                    f"fedds selects {per_round} client(s) per round, but Dawid-Skene "
                    "needs at least two annotators; raise fedds.frac or "
                    "fedds.frac_by_num_clients"
                )

        name = str(self.runtime.exp_name)
        if not name.strip():
            raise ValueError("runtime.exp_name must not be empty")
        if any(sep in name for sep in ("/", "\\", "..")):
            raise ValueError(
                f"runtime.exp_name '{name}' must be a plain name (no path separators)"
            )
        self.runtime.exp_name = name
        if self.runtime.save_every < 0:
            raise ValueError("runtime.save_every must be >= 0")
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers must be >= 0")
        _require_seed(self.runtime.seed, "runtime.seed")

    # -------------------------------------------------------------- utilities
    def annotators_per_round(self) -> int:
        return self.fedds.clients_per_round(self.data.num_clients)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    def run_dir(self) -> str:
        return os.path.join(self.runtime.output_root, self.runtime.exp_name)


def _require_int(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _finite_positive(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value))) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _require_seed(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _as_list(value: Any) -> List[Any]:
    """Accept a real list or a comma-separated string from the CLI."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_frac_map(value: Any) -> Dict[int, float]:
    """Accept {"10": 0.3}, "10:0.3,50:0.1", or ["10:0.3", "50:0.1"]."""
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    items: List[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value]
    pairs = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"cannot parse frac_by_num_clients entry: {item!r}")
        key, raw_value = item.split(":", 1)
        pairs[int(key)] = float(raw_value)
    if not pairs:
        raise ValueError("frac_by_num_clients is empty or unparsable")
    return pairs


def _read_structured(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "PyYAML is not installed; use a .json config or `pip install pyyaml`."
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    raise ValueError(f"unsupported config extension: {ext}")


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_cli(argv: List[str]) -> Tuple[str, Dict[str, Any]]:
    """`python run.py config.json --data.num_clients=10 --fedds.rounds=20`."""
    if not argv:
        raise SystemExit("usage: run.py <config.(json|yaml)> [--section.key=value ...]")
    path, rest = argv[0], argv[1:]
    overrides: Dict[str, Any] = {}
    for token in rest:
        if not token.startswith("--") or "=" not in token:
            raise SystemExit(f"cannot parse override: {token}")
        key, value = token[2:].split("=", 1)
        section, _, name = key.partition(".")
        if not name:
            raise SystemExit(f"override needs section.key form: {token}")
        overrides.setdefault(section, {})[name] = _coerce(value)
    return path, overrides


def _coerce(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [_coerce(part.strip()) for part in text.split(",")]
    return text
