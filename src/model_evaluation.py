"""Multi-metric evaluation. Never fabricates scores."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

from src.config import CLASS_NAMES


def false_positive_rate(y_true, y_pred) -> float | None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    if denom == 0:
        return None
    return float(fp / denom)


def evaluate_classifier(
    pipeline,
    X,
    y,
    split_name: str,
    data_are_synthetic: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
        except Exception:
            proba = None
    y_pred = pipeline.predict(X)
    infer_s = time.perf_counter() - t0

    y_true = np.asarray(y)
    y_hat = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    scores: dict[str, Any] = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "false_positive_rate": false_positive_rate(y_true, y_hat),
        "confusion_matrix": {
            "labels": [CLASS_NAMES[0], CLASS_NAMES[1]],
            "matrix": cm.astype(int).tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "inference_seconds": float(infer_s),
        "source": (
            "synthetic_demo_split"
            if data_are_synthetic
            else "user_dataset_split"
        ),
        "disclaimer": (
            "These figures were computed on the current local split. "
            "They are not copied from published papers and are not comparable "
            "to results from different datasets or protocols."
        ),
    }

    if proba is not None and proba.shape[1] >= 2:
        y_score = proba[:, 1]
        if len(np.unique(y_true)) == 2:
            scores["roc_auc"] = float(roc_auc_score(y_true, y_score))
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            scores["pr_auc"] = float(auc(rec, prec))
        else:
            scores["roc_auc"] = None
            scores["pr_auc"] = None
        scores["has_probabilities"] = True
    else:
        scores["roc_auc"] = None
        scores["pr_auc"] = None
        scores["has_probabilities"] = False

    return scores
