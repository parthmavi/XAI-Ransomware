"""Score tabular samples. Does not execute files or scan binaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, FEATURE_COLUMNS


def predict_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    features = X.loc[:, list(FEATURE_COLUMNS)]
    y_hat = pipeline.predict(features)
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(features)
        except Exception:
            proba = None
    rows = []
    for i, pred in enumerate(np.asarray(y_hat)):
        pred_int = int(pred)
        row: dict[str, Any] = {
            "row_index": int(i),
            "predicted_class_id": pred_int,
            "predicted_class": CLASS_NAMES.get(pred_int, str(pred_int)),
        }
        if proba is not None:
            p_benign = float(proba[i, 0])
            p_ransom = float(proba[i, 1])
            row["p_benign"] = p_benign
            row["p_ransomware"] = p_ransom
            row["model_score_ransomware"] = p_ransom
            row["score_label"] = "Predicted probability of class ransomware (predict_proba[:, 1])"
        else:
            row["p_benign"] = None
            row["p_ransomware"] = None
            row["model_score_ransomware"] = None
            row["score_label"] = "This estimator did not return class probabilities."
        rows.append(row)
    return pd.DataFrame(rows)


def analyst_summary(pred_row: dict[str, Any], top_features: list[tuple[str, float]] | None) -> str:
    label = pred_row.get("predicted_class", "unknown")
    score = pred_row.get("p_ransomware")
    score_txt = (
        f"The model assigned a ransomware-class probability of {score:.3f} "
        "(this is a model score, not a measured infection likelihood)."
        if isinstance(score, float)
        else "Class probabilities were not available for this estimator."
    )
    feat_txt = ""
    if top_features:
        bits = [f"{name} (attribution {value:+.4f})" for name, value in top_features[:5]]
        feat_txt = " Largest-magnitude attributions for this row: " + "; ".join(bits) + "."
    return (
        f"Predicted class: {label}. {score_txt}{feat_txt} "
        "Attributions describe this trained model on this feature table; they do not "
        "prove that a process is ransomware, and this application does not execute samples."
    )
