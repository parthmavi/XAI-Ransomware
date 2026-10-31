"""Unit tests for schema validation, splitting, and a tiny training run."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS
from src.data_preprocessing import encode_labels, leakage_aware_split
from src.feature_extraction import extract_feature_frame, inspect_columns
from src.model_evaluation import evaluate_classifier
from src.model_training import train_model
from src.prediction import predict_frame
from src.security import UploadValidationError, validate_upload


def _tiny_frame(n: int = 40) -> pd.DataFrame:
    data = {c: [float(i % 7) for i in range(n)] for c in FEATURE_COLUMNS}
    data["label"] = ["benign" if i % 2 == 0 else "ransomware" for i in range(n)]
    data["data_source"] = ["SYNTHETIC_DEMO"] * n
    data["sample_id"] = [f"t-{i}" for i in range(n)]
    data["unknown_extra"] = list(range(n))
    return pd.DataFrame(data)


def test_reject_exe() -> None:
    with pytest.raises(UploadValidationError):
        validate_upload("payload.exe", b"MZ")


def test_accept_csv() -> None:
    name = validate_upload("table.csv", b"a,b\n1,2\n")
    assert name == "table.csv"


def test_schema_ignores_unknown() -> None:
    report = inspect_columns(_tiny_frame().columns)
    assert "unknown_extra" in report.ignored_columns
    assert report.has_label
    assert report.missing_required == []


def test_extract_and_encode() -> None:
    table, report = extract_feature_frame(_tiny_frame(), require_label=True)
    assert report.is_synthetic_demo
    y = encode_labels(table["label"])
    assert set(y.unique()) == {0, 1}


def test_split_and_train() -> None:
    table, _ = extract_feature_frame(_tiny_frame(60), require_label=True)
    split = leakage_aware_split(table)
    trained = train_model("random_forest", split.X_train, split.y_train)
    metrics = evaluate_classifier(
        trained.pipeline, split.X_test, split.y_test, "test", True
    )
    assert "accuracy" in metrics
    assert metrics["source"] == "synthetic_demo_split"
    preds = predict_frame(trained.pipeline, split.X_test.head(3))
    assert "predicted_class" in preds.columns
