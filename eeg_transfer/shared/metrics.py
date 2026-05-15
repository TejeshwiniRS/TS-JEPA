"""Metrics matching EEG-FM-Bench (Appendix B.4).

For multi-class classification (BCIC-IV 2a, 4-class), the bench reports:
  - Balanced Accuracy  (primary)
  - Weighted F1
  - Cohen's Kappa
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "bacc":  float(balanced_accuracy_score(y_true, y_pred)),
        "wf1":   float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
