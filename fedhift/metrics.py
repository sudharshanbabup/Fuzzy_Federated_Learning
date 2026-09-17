"""Evaluation metrics: global accuracy, per-client fairness, detection quality."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, batch: int = 512) -> tuple[float, float]:
    model.eval()
    correct, loss_sum, n = 0, 0.0, len(y)
    lf = torch.nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, n, batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        out = model(xb)
        loss_sum += lf(out, yb).item()
        correct += (out.argmax(1) == yb).sum().item()
    return correct / max(n, 1), loss_sum / max(n, 1)


@torch.no_grad()
def evaluate_detailed(model, x: torch.Tensor, y: torch.Tensor, num_classes: int,
                      batch: int = 512) -> dict:
    """Plain accuracy, balanced accuracy and macro F1 on one evaluation set.

    Balanced accuracy is the mean per-class recall and macro F1 the unweighted
    mean of the per-class F1 scores, both computed over the classes that are
    present in ``y``. Reporting these alongside plain accuracy separates a
    genuine fairness effect from the differing difficulty of the per-client
    evaluation sets, which under label skew are not comparable to each other.
    """
    model.eval()
    n = len(y)
    if n == 0:
        return {"acc": float("nan"), "bacc": float("nan"), "macro_f1": float("nan")}
    preds = []
    for i in range(0, n, batch):
        preds.append(model(x[i:i + batch]).argmax(1))
    p = torch.cat(preds)
    acc = float((p == y).sum().item()) / n
    recalls, f1s = [], []
    for c in range(num_classes):
        support = (y == c)
        ns = int(support.sum().item())
        if ns == 0:
            continue
        tp = int((p[support] == c).sum().item())
        rec = tp / ns
        pred_c = int((p == c).sum().item())
        prec = tp / pred_c if pred_c else 0.0
        recalls.append(rec)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {"acc": acc,
            "bacc": float(np.mean(recalls)) if recalls else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan")}


def jain_index(a: Sequence[float]) -> float:
    """Jain's fairness index: 1 = perfectly uniform service across clients."""
    a = np.asarray(a, dtype=np.float64)
    s = a.sum()
    if s <= 0:
        return 0.0
    return float(s ** 2 / (len(a) * (a ** 2).sum()))


def worst_frac(a: Sequence[float], frac: float = 0.1) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    k = max(1, int(np.ceil(frac * len(a))))
    return float(a[:k].mean())


def auc_binary(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC of a detector; labels 1 = benign, 0 = malicious.

    Computed from the rank statistic, so it is exact and ties are handled.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for u in np.where(cnt > 1)[0]:
        m = inv == u
        ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
