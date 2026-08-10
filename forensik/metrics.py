"""Evaluation metrics.

Throughout, ``y_true`` uses 1 for fake and 0 for real, and ``y_score`` is the
predicted probability of fake.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def equal_error_rate(y_true, y_score) -> tuple[float, float]:
    """EER and the threshold where false accept and false reject meet."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])


def threshold_from_validation(y_true, y_score, criterion: str = "youden") -> float:
    """Pick a threshold on validation data. Never call this on the test set."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if criterion == "youden":
        return float(thresholds[int(np.argmax(tpr - fpr))])
    if criterion == "eer":
        return equal_error_rate(y_true, y_score)[1]
    if criterion == "f1":
        best_f1, best_threshold = -1.0, 0.5
        for t in thresholds:
            pred = np.asarray(y_score) >= t
            tp = int((pred & (y_true == 1)).sum())
            fp = int((pred & (y_true == 0)).sum())
            fn = int((~pred & (y_true == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(t)
        return best_threshold
    raise ValueError(f"unknown criterion: {criterion}")


def prior_matched_threshold(y_score, positive_rate: float = 0.5) -> float:
    """Threshold that makes the predicted positive rate match a known prior.

    Uses the score distribution and the class balance, never the test labels.
    Report it separately from a fixed threshold and label it as transductive.
    """
    return float(np.quantile(np.asarray(y_score, dtype=float), 1.0 - positive_rate))


def expected_calibration_error(y_true, y_prob, bins: int = 15) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins."""
    confidence = np.maximum(y_prob, 1 - y_prob)
    correct = ((y_prob >= 0.5).astype(int) == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def full_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    accuracy = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    eer, eer_threshold = equal_error_rate(y_true, y_score)

    return {
        "n": int(n),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "auc": auc,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_errors": int(fp + fn),
        "ci95_pp": float(1.96 * np.sqrt(accuracy * (1 - accuracy) / n) * 100),
        "ece": expected_calibration_error(y_true, y_score),
    }


def mcnemar(y_true, pred_a, pred_b, exact_below: int = 25) -> dict:
    """Paired test for two models scored on the same test set."""
    y_true = np.asarray(y_true).astype(int)
    a_right = np.asarray(pred_a).astype(int) == y_true
    b_right = np.asarray(pred_b).astype(int) == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0, "test": "degenerate"}

    if n < exact_below:
        from math import comb

        tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
        return {
            "only_a": only_a,
            "only_b": only_b,
            "p_value": float(min(tail * 2 / 2**n, 1.0)),
            "test": "exact",
        }

    from scipy.stats import chi2

    statistic = (abs(only_a - only_b) - 1) ** 2 / n
    return {
        "only_a": only_a,
        "only_b": only_b,
        "p_value": float(1 - chi2.cdf(statistic, 1)),
        "chi2": float(statistic),
        "test": "chi2",
    }


class TemperatureScaler:
    """Single-parameter calibration, fitted on validation logits."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits, y_true, lr: float = 0.01, steps: int = 300):
        import torch

        values = torch.tensor(np.asarray(logits), dtype=torch.float32)
        target = torch.tensor(np.asarray(y_true), dtype=torch.long)
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([log_t], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(values / log_t.exp(), target)
            loss.backward()
            optimizer.step()
        self.temperature = float(log_t.exp().item())
        return self

    def transform(self, logits) -> np.ndarray:
        scaled = np.asarray(logits, dtype=float) / self.temperature
        exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
