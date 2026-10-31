"""LIME local explanations. Stochastic; seed is fixed but results can still vary."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, FEATURE_COLUMNS, RANDOM_STATE
from src.data_preprocessing import transform_frame


class LimeUnavailable(RuntimeError):
    pass


def explain_row(
    pipeline,
    X_background: pd.DataFrame,
    X_row: pd.DataFrame,
    row_index: int = 0,
    num_features: int = 10,
) -> dict[str, Any]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise LimeUnavailable("lime is not installed.") from exc

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    bg = transform_frame(preprocess, X_background)
    row = transform_frame(preprocess, X_row.iloc[[row_index]])
    names = list(bg.columns)

    explainer = LimeTabularExplainer(
        training_data=bg.values.astype(float),
        feature_names=names,
        class_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
        mode="classification",
    )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        return model.predict_proba(data)

    t0 = time.perf_counter()
    explanation = explainer.explain_instance(
        data_row=row.values.astype(float)[0],
        predict_fn=predict_fn,
        num_features=num_features,
        top_labels=2,
    )
    elapsed = time.perf_counter() - t0
    label = 1 if 1 in explanation.available_labels() else explanation.available_labels()[0]
    weights = explanation.as_list(label=label)
    intercept = None
    try:
        intercept = float(explanation.intercept[label])
    except Exception:
        intercept = None
    local_pred = None
    try:
        local_pred = float(explanation.local_pred[0]) if hasattr(explanation.local_pred, "__len__") else float(explanation.local_pred)
    except Exception:
        local_pred = None

    frame = pd.DataFrame(weights, columns=["rule", "weight"])
    return {
        "label_explained": int(label),
        "class_name": CLASS_NAMES.get(int(label), str(label)),
        "weights": frame,
        "intercept": intercept,
        "local_prediction": local_pred,
        "seconds": float(elapsed),
        "method_note": (
            "LIME fits a local linear surrogate on perturbed samples. "
            "Explanations can change between runs even with a fixed seed. "
            "They describe the model, not a verified cause of ransomware behaviour."
        ),
    }
