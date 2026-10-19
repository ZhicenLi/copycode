"""FedCorr's federated pre-processing stage: identify, estimate, correct.

Faithfulness notes
------------------
1. **Chain semantics.** FedCorr aggregates ``int(K * frac1)`` clients per step
   and ``frac1 = 1/K``, so each step aggregates exactly one client. `FedAvg`
   over a single client returns that client's weights unchanged, which means the
   "global" model is carried from one client to the next instead of being
   averaged. This class reproduces that behaviour on purpose: changing it would
   mean running a different algorithm from the one that produced the published
   numbers.
2. **One iteration visits every client exactly once** (sampling without
   replacement), then runs the GMM grouping and one correction pass.
3. **Recognition and correction alternate** at the end of every iteration; they
   are not two separate phases.
4. Only the corrected labels are exported. Stages 2 and 3 of the original paper
   are not run, so the second correction pass that follows stage 2 is not
   applied either.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..core.gmm import noisy_mask, split_two_components
from ..core.lid import lid_scores
from ..data.loaders import build_transform, evaluate, make_loader
from ..models.registry import build_model
from ..seeding import derive_seed
from ..seeding import rng as make_rng


@dataclass
class FedCorrResult:
    corrected_labels: np.ndarray
    per_iteration: List[Dict] = field(default_factory=list)
    per_client: List[Dict] = field(default_factory=list)


class FedCorrPreprocessor:
    def __init__(
        self,
        cfg: Config,
        manifest,
        train_images: torch.Tensor,
        test_loader,
        device: torch.device,
        output_dir: str,
    ):
        self.cfg = cfg
        self.manifest = manifest
        self.images = train_images
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.spec = cfg.data.spec()
        self.num_clients = cfg.data.num_clients
        self.labels = manifest.noisy_label.copy()      # working copy, corrected in place
        self.client_indices = {
            client: manifest.client_indices(client) for client in range(self.num_clients)
        }

        self.model = self._new_model()
        self.local = self._new_model()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        self._train_tf = build_transform(self.spec, "train", cfg.fedcorr.preprocess)
        self._eval_tf = build_transform(self.spec, "eval", cfg.fedcorr.preprocess)
        self._score_tf = (
            self._train_tf if cfg.fedcorr.train_transform == "train" else self._eval_tf
        )

        self.iteration_records: List[Dict] = []
        self.client_records: List[Dict] = []

    def _new_model(self) -> nn.Module:
        return build_model(
            self.cfg.model.fedcorr, self.cfg.data.num_classes, self.spec.channels
        ).to(self.device)

    # --------------------------------------------------------------- training
    def _local_train(
        self, indices: np.ndarray, anchor: List[torch.Tensor], mu: float, seed: int
    ) -> float:
        """One client's local update, following FedCorr's `update_weights`."""
        cfg = self.cfg.fedcorr
        # global RNGs are consumed by the augmentation pipeline; seeding per client
        # makes the update reproducible regardless of what ran before it
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
        optimizer = torch.optim.SGD(self.local.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        self.local.train()

        # local streams, so a client's update is reproducible on its own seed
        mix_rng = np.random.default_rng(derive_seed(seed, "mixup"))
        perm_generator = torch.Generator().manual_seed(derive_seed(seed, "mixup_perm"))

        epoch_losses: List[float] = []
        for _ in range(cfg.local_ep):
            batch_losses: List[float] = []
            for batch_index, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                optimizer.zero_grad()

                if cfg.mixup:
                    lam = float(mix_rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(images.size(0), generator=perm_generator).to(images.device)
                    mixed = lam * images + (1.0 - lam) * images[perm]
                    logits = self.local(mixed)
                    loss = lam * self.loss_fn(logits, labels).mean() + \
                        (1.0 - lam) * self.loss_fn(logits, labels[perm]).mean()
                else:
                    logits = self.local(images)
                    loss = self.loss_fn(logits, labels).mean()

                # FedCorr adds the proximal term from the second batch onwards
                if cfg.beta > 0 and batch_index > 0 and mu > 0:
                    diff = torch.zeros((), device=self.device)
                    for parameter, reference in zip(self.local.parameters(), anchor):
                        diff = diff + torch.pow(torch.norm(reference - parameter), 2)
                    loss = loss + cfg.beta * mu * torch.sqrt(diff)

                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def _score(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Softmax probabilities and per-sample loss, in `indices` order."""
        loader = make_loader(
            self.images, self.labels, indices,
            batch_size=self.cfg.fedcorr.local_bs, train=False, transform=self._score_tf,
            num_workers=0,
        )
        self.model.eval()
        probs: List[np.ndarray] = []
        losses: List[np.ndarray] = []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).long()
            logits = self.model(images)
            probability = torch.softmax(logits, dim=1)
            if self.cfg.fedcorr.loss_on == "prob":
                # the upstream code feeds probabilities into CrossEntropyLoss
                batch_loss = self.loss_fn(probability, labels)
            else:
                batch_loss = self.loss_fn(logits, labels)
            probs.append(probability.cpu().numpy())
            losses.append(batch_loss.cpu().numpy())
        if not probs:
            return np.zeros((0, self.cfg.data.num_classes)), np.zeros(0)
        return np.concatenate(probs), np.concatenate(losses)

    # -------------------------------------------------------------------- run
    def run(self) -> FedCorrResult:
        cfg = self.cfg.fedcorr
        num_clients = self.num_clients
        start = time.time()

        # start from clean logs; appending to a previous run would mix records.
        # old checkpoints are removed too: they are never loaded and would pile up
        for name in ("fedcorr_clients.jsonl", "fedcorr_iterations.jsonl"):
            path = os.path.join(self.output_dir, name)
            if os.path.exists(path):
                os.remove(path)
        for stale in glob.glob(os.path.join(self.output_dir, "fedcorr_ckpt_*.pt")):
            os.remove(stale)

        lid_accumulated = np.zeros(num_clients, dtype=np.float64)
        estimated_noise = np.zeros(num_clients, dtype=np.float64)

        for iteration in range(cfg.iteration1):
            lid_this_iteration = np.zeros(num_clients, dtype=np.float64)
            loss_by_sample = np.zeros(len(self.labels), dtype=np.float64)
            order = make_rng(
                self.cfg.runtime.seed, "fedcorr_order", iteration
            ).permutation(num_clients)

            for step, client in enumerate(order):
                client = int(client)
                indices = self.client_indices[client]

                self.local.load_state_dict(self.model.state_dict())
                anchor = [p.detach().clone() for p in self.local.parameters()]

                mu = float(estimated_noise[client]) if iteration > 0 else 0.0
                seed = int(
                    make_rng(self.cfg.runtime.seed, "fedcorr_local", iteration, client)
                    .integers(0, 2 ** 31 - 1)
                )
                local_loss = self._local_train(indices, anchor, mu, seed)

                # one client per aggregation step -> the global model becomes this client's
                self.model.load_state_dict(self.local.state_dict())

                probs, losses = self._score(indices)
                lid_values = (
                    lid_scores(probs, k=cfg.lid_k) if len(indices) > 1 else np.zeros(len(indices))
                )
                lid_this_iteration[client] = float(np.mean(lid_values))
                loss_by_sample[indices] = losses

                record = {
                    "iteration": iteration,
                    "step": step,
                    "client": client,
                    "samples": int(len(indices)),
                    "local_loss": local_loss,
                    "lid_mean": float(lid_this_iteration[client]),
                    "mu": mu,
                    "weights_sha": _digest_state(self.model.state_dict()),
                }
                self.client_records.append(record)
                _append_jsonl(os.path.join(self.output_dir, "fedcorr_clients.jsonl"), record)

                if (
                    self.cfg.runtime.save_every
                    and (step + 1) % self.cfg.runtime.save_every == 0
                ):
                    self._checkpoint(iteration, step + 1, lid_accumulated, estimated_noise)

            lid_accumulated = lid_accumulated + lid_this_iteration

            # ---- client-level split over the accumulated LID scores
            split = split_two_components(lid_accumulated, random_state=self.cfg.runtime.seed)
            noisy_clients = np.where(noisy_mask(split))[0]

            # ---- per-sample noise level inside each noisy client
            estimated_noise = np.zeros(num_clients, dtype=np.float64)
            for client in noisy_clients:
                indices = self.client_indices[int(client)]
                if len(indices) < 2:
                    continue
                loss_split = split_two_components(
                    loss_by_sample[indices], random_state=self.cfg.runtime.seed
                )
                estimated_noise[client] = float(np.mean(noisy_mask(loss_split)))

            # ---- label correction (faithful to the upstream lines 162-175)
            corrected_this_iteration = 0
            if cfg.correct:
                for client in noisy_clients:
                    indices = self.client_indices[int(client)]
                    if len(indices) == 0:
                        continue
                    probs_global, _ = self._score(indices)
                    budget = int(len(indices) * estimated_noise[client] * cfg.relabel_ratio)
                    if budget <= 0:
                        continue
                    top_loss = np.argsort(-loss_by_sample[indices])[:budget]
                    confident = np.where(probs_global.max(axis=1) > cfg.confidence_thres)[0]
                    chosen = np.intersect1d(top_loss, confident)
                    if len(chosen) == 0:
                        continue
                    new_labels = probs_global.argmax(axis=1)[chosen]
                    previous = self.labels[indices[chosen]]
                    # count real changes, not rewrite attempts: argmax often equals
                    # the current label, and the same sample can be revisited later
                    corrected_this_iteration += int(np.sum(new_labels != previous))
                    self.labels[indices[chosen]] = new_labels

            accuracy = evaluate(self.model, self.test_loader, self.device)
            record = {
                "iteration": iteration,
                "clients_visited": int(len(order)),
                "noisy_clients": int(len(noisy_clients)),
                "noisy_client_ids": [int(c) for c in noisy_clients],
                "estimated_noise": [float(v) for v in estimated_noise],
                "lid_accumulated": [float(v) for v in lid_accumulated],
                "corrected_samples": corrected_this_iteration,
                "total_corrected": int(np.sum(self.labels != self.manifest.noisy_label)),
                "estimated_noise_mean": float(estimated_noise.mean()),
                "test_accuracy": accuracy,
                "gmm_backend": split.backend,
                "elapsed_s": round(time.time() - start, 1),
            }
            self.iteration_records.append(record)
            _append_jsonl(os.path.join(self.output_dir, "fedcorr_iterations.jsonl"), record)
            self._checkpoint(iteration + 1, 0, lid_accumulated, estimated_noise)

        self._save_result()
        return FedCorrResult(
            corrected_labels=self.labels.copy(),
            per_iteration=self.iteration_records,
            per_client=self.client_records,
        )

    # ------------------------------------------------------------ persistence
    def _checkpoint(self, iteration: int, step: int, lid, noise) -> None:
        if not self.cfg.runtime.checkpoint_models:
            return
        path = os.path.join(self.output_dir, f"fedcorr_ckpt_i{iteration}_s{step}.pt")
        torch.save({
            "iteration": iteration,
            "step": step,
            "model": self.model.state_dict(),
            "lid_accumulated": lid,
            "estimated_noise": noise,
            "labels": self.labels,
        }, path)

    def _save_result(self) -> None:
        np.save(os.path.join(self.output_dir, "corrected_labels.npy"), self.labels)
        summary = {
            "iterations": self.cfg.fedcorr.iteration1,
            "clients": self.num_clients,
            "noisy_at_last_iteration": (
                self.iteration_records[-1]["noisy_clients"] if self.iteration_records else 0
            ),
            "corrected_samples": int(np.sum(self.labels != self.manifest.noisy_label)),
            "accuracy_curve": [r["test_accuracy"] for r in self.iteration_records],
        }
        with open(os.path.join(self.output_dir, "fedcorr_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)


def _append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _digest_state(state: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode("utf-8"))
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]
