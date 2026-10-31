"""SHAP global and local explanations for the fitted sklearn pipeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, RANDOM_STATE
from src.data_preprocessing import transform_frame


class ShapUnavailable(RuntimeError):
    pass


def _model_and_transformed(pipeline, X: pd.DataFrame):
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    Xt = transform_frame(preprocess, X)
    return model, Xt


def _tree_explainer(model):
    import shap

    return shap.TreeExplainer(model)


def shap_values_for(pipeline, X: pd.DataFrame, background: pd.DataFrame | None = None) -> dict[str, Any]:
    try:
        import shap
    except ImportError as exc:
        raise ShapUnavailable("shap is not installed.") from exc

    model, Xt = _model_and_transformed(pipeline, X)
    t0 = time.perf_counter()
    kind = type(model).__name__
    try:
        if kind in {"RandomForestClassifier", "XGBClassifier", "XGBModel"}:
            explainer = _tree_explainer(model)
            sv = explainer.shap_values(Xt.values)
            expected = explainer.expected_value
        else:
            bg = background if background is not None else X
            _, bg_t = _model_and_transformed(pipeline, bg)
            if len(bg_t) > 50:
                bg_t = bg_t.sample(n=50, random_state=RANDOM_STATE)
            explainer = shap.KernelExplainer(
                lambda data: model.predict_proba(data)[:, 1],
                bg_t.values,
            )
            sv = explainer.shap_values(Xt.values, nsamples=100)
            expected = explainer.expected_value
    except Exception as exc:
        raise ShapUnavailable(f"SHAP could not explain this estimator: {exc}") from exc

    elapsed = time.perf_counter() - t0
    values = _positive_class_shap(sv)
    names = list(Xt.columns) if len(Xt.columns) == values.shape[1] else list(FEATURE_COLUMNS)
    return {
        "feature_names": names,
        "values": np.asarray(values, dtype=float),
        "expected_value": _as_float_expected(expected),
        "seconds": float(elapsed),
        "method_note": (
            "TreeExplainer was used for tree ensembles; KernelExplainer otherwise. "
            "SHAP attributes the model's output, not a causal ransomware mechanism."
        ),
    }


def _positive_class_shap(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[1] if len(sv) > 1 else sv[0], dtype=float)
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[0] == 2:
            return arr[1]
    return arr


def _as_float_expected(expected) -> float:
    if isinstance(expected, (list, tuple, np.ndarray)):
        seq = np.asarray(expected, dtype=float).ravel()
        return float(seq[-1] if len(seq) else 0.0)
    return float(expected)


def global_importance(shap_pack: dict[str, Any]) -> pd.DataFrame:
    values = np.abs(shap_pack["values"]).mean(axis=0)
    names = shap_pack["feature_names"]
    frame = pd.DataFrame({"feature": names, "mean_abs_shap": values})
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_contributions(shap_pack: dict[str, Any], row_index: int) -> pd.DataFrame:
    values = shap_pack["values"][row_index]
    names = shap_pack["feature_names"]
    frame = pd.DataFrame({"feature": names, "shap_value": values})
    frame["abs_shap"] = frame["shap_value"].abs()
    return frame.sort_values("abs_shap", ascending=False).reset_index(drop=True)
