"""Ground-truth diagnostics for the FedCorr stage.

The training pipeline never sees clean labels. This module is the only place
that does, and it runs *after* the fact to answer the questions a paper must
answer:

* client identification: did the LID/GMM split separate the noisy clients?
* noise estimation: how close is the per-client estimate to the truth?
* correction: when a label was rewritten, was a wrong one fixed or a right one
  broken?

Everything here is read-only analysis; nothing feeds back into training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .core.dawid_skene import spearman


def true_client_noise(manifest) -> np.ndarray:
    """Realised flip ratio per client (not the requested rate)."""
    rates = np.zeros(int(manifest.meta.get("num_clients", 0)), dtype=np.float64)
    for client in range(len(rates)):
        indices = manifest.client_indices(client)
        if len(indices) == 0:
            continue
        rates[client] = float(np.mean(manifest.noisy_label[indices] != manifest.clean_label[indices]))
    return rates


def identification_metrics(
    estimated: Sequence[float], truth: Sequence[float]
) -> Dict[str, float]:
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    predicted_noisy = estimated > 0.0
    actually_noisy = truth > 0.0

    tp = int(np.sum(predicted_noisy & actually_noisy))
    fp = int(np.sum(predicted_noisy & ~actually_noisy))
    fn = int(np.sum(~predicted_noisy & actually_noisy))
    tn = int(np.sum(~predicted_noisy & ~actually_noisy))

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    accuracy = (tp + tn) / max(len(truth), 1)
    return {
        "spearman_estimated_vs_true": float(spearman(estimated, truth)),
        "mean_absolute_error": float(np.mean(np.abs(estimated - truth))),
        "client_precision": float(precision),
        "client_recall": float(recall),
        "client_accuracy": float(accuracy),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def correction_metrics(manifest, corrected: np.ndarray) -> Dict[str, float]:
    """Compare rewritten labels with the clean ground truth.

    Only client samples are counted: the public split never enters training and
    is never corrected, so including it would dilute the number.
    """
    client_mask = manifest.split == "client"
    clean = manifest.clean_label[client_mask]
    noisy = manifest.noisy_label[client_mask]
    corrected = np.asarray(corrected)[client_mask]

    changed = corrected != noisy
    was_wrong = noisy != clean
    now_right = corrected == clean

    fixed = changed & was_wrong & now_right
    still_wrong = changed & was_wrong & ~now_right
    broke = changed & ~was_wrong

    total_wrong = int(np.sum(was_wrong))
    n_changed = int(np.sum(changed))
    return {
        "label_accuracy_before": float(np.mean(noisy == clean)),
        "label_accuracy_after": float(np.mean(corrected == clean)),
        "noisy_labels_total": total_wrong,
        "labels_changed": n_changed,
        "fixed": int(np.sum(fixed)),
        "still_wrong_after_rewrite": int(np.sum(still_wrong)),
        "broken": int(np.sum(broke)),
        "correction_precision": float(np.sum(fixed) / n_changed) if n_changed else float("nan"),
        "correction_recall": float(np.sum(fixed) / total_wrong) if total_wrong else float("nan"),
    }


def summarise(
    manifest,
    corrected: np.ndarray,
    iteration_records: List[Dict],
) -> Dict[str, object]:
    """One JSON-serialisable block for the paper's analysis section."""
    truth = true_client_noise(manifest)

    last = iteration_records[-1] if iteration_records else {}
    estimated = last.get("estimated_noise")
    analysis: Dict[str, object] = {
        "clients": int(len(truth)),
        "true_noise_mean": float(truth.mean()),
        "clients_with_noise": int(np.sum(truth > 0)),
        "correction": correction_metrics(manifest, corrected),
    }
    if estimated is not None:
        analysis["identification"] = identification_metrics(estimated, truth)
    else:
        analysis["identification"] = None

    analysis["per_client"] = [
        {
            "client_id": int(client),
            "true_noise": float(truth[client]),
            "estimated_noise": float(estimated[client]) if estimated is not None else None,
        }
        for client in range(len(truth))
    ]
    return analysis


def format_table(analysis: Dict[str, object]) -> str:
    correction = analysis["correction"]  # type: ignore[index]
    lines = [
        "FedCorr 诊断",
        f"  标签正确率   {correction['label_accuracy_before']:.4f} -> "
        f"{correction['label_accuracy_after']:.4f}",
        f"  改写样本     {correction['labels_changed']}"
        f"（改对 {correction['fixed']} / 改错 {correction['broken'] + correction['still_wrong_after_rewrite']}）",
        f"  纠错精确率   {correction['correction_precision']:.4f}",
        f"  纠错召回率   {correction['correction_recall']:.4f}",
    ]
    identification = analysis.get("identification")
    if identification:
        lines.extend([
            f"  客户端识别   准确率 {identification['client_accuracy']:.4f} "
            f"(TP {identification['tp']} / FP {identification['fp']})",
            f"  噪声估计     与真值 Spearman {identification['spearman_estimated_vs_true']:.4f}",
        ])
    return "\n".join(lines)
