"""Orchestrates modules without putting logic inside the Streamlit page."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from src.config import FEATURE_COLUMNS, LABEL_COLUMN
from src.data_preprocessing import class_balance, encode_labels, leakage_aware_split
from src.database import record_prediction
from src.feature_extraction import extract_feature_frame
from src.lime_explainer import LimeUnavailable, explain_row
from src.model_evaluation import evaluate_classifier
from src.model_training import persist_model, train_model
from src.prediction import analyst_summary, predict_frame
from src.security import validate_upload
from src.shap_explainer import (
    ShapUnavailable,
    global_importance,
    local_contributions,
    shap_values_for,
)


def read_csv_bytes(filename: str, raw: bytes) -> pd.DataFrame:
    validate_upload(filename, raw)
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"CSV could not be parsed: {exc}") from exc


def load_feature_table(df: pd.DataFrame, require_label: bool) -> tuple[pd.DataFrame, Any]:
    return extract_feature_frame(df, require_label=require_label)


def train_and_evaluate(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    table, report = extract_feature_frame(df, require_label=True)
    split = leakage_aware_split(table)
    trained = train_model(model_name, split.X_train, split.y_train, split.X_valid, split.y_valid)
    trained.n_test = int(len(split.X_test))
    is_demo = report.is_synthetic_demo
    valid_metrics = evaluate_classifier(
        trained.pipeline, split.X_valid, split.y_valid, "validation", is_demo
    )
    test_metrics = evaluate_classifier(
        trained.pipeline, split.X_test, split.y_test, "test", is_demo
    )
    y_all = encode_labels(table[LABEL_COLUMN])
    bundle = {
        "trained": trained,
        "report": report,
        "split": split,
        "balance": class_balance(y_all),
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "is_synthetic_demo": is_demo,
    }
    persist_model(
        trained,
        evaluation={
            "model": model_name,
            "validation": valid_metrics,
            "test": test_metrics,
            "source": test_metrics["source"],
            "split_notes": split.notes,
            "is_synthetic_demo": is_demo,
        },
    )
    return bundle


def detect_and_explain(
    pipeline,
    model_name: str,
    X: pd.DataFrame,
    background: pd.DataFrame,
    source_name: str,
    is_synthetic: bool,
    row_index: int = 0,
    persist_history: bool = True,
) -> dict[str, Any]:
    preds = predict_frame(pipeline, X)
    shap_pack = None
    shap_error = None
    lime_pack = None
    lime_error = None
    top_pairs: list[tuple[str, float]] = []
    try:
        shap_pack = shap_values_for(pipeline, X.iloc[[row_index]], background=background)
        local = local_contributions(shap_pack, 0)
        top_pairs = list(zip(local["feature"].tolist(), local["shap_value"].tolist()))
        shap_pack["local"] = local
        shap_pack["global_on_batch"] = global_importance(
            shap_values_for(pipeline, X if len(X) <= 80 else X.head(80), background=background)
        )
    except ShapUnavailable as exc:
        shap_error = str(exc)
    try:
        lime_pack = explain_row(pipeline, background, X, row_index=row_index)
    except LimeUnavailable as exc:
        lime_error = str(exc)

    pred_row = preds.iloc[row_index].to_dict()
    summary = analyst_summary(pred_row, top_pairs or None)
    if persist_history:
        payload = X.iloc[row_index][list(FEATURE_COLUMNS)].to_dict()
        record_prediction(
            model_name=model_name,
            predicted_class=pred_row["predicted_class"],
            p_ransomware=pred_row.get("p_ransomware"),
            source_name=source_name,
            is_synthetic=is_synthetic,
            row_payload=payload,
            explanation_summary=summary,
        )
    return {
        "predictions": preds,
        "selected": pred_row,
        "shap": shap_pack,
        "shap_error": shap_error,
        "lime": lime_pack,
        "lime_error": lime_error,
        "analyst_summary": summary,
    }
