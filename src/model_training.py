"""Train Random Forest, XGBoost, and optional SVM on identical splits."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    ACTIVE_MODEL_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    SCHEMA_PATH,
    ensure_runtime_dirs,
)
from src.data_preprocessing import build_preprocessor

ModelName = Literal["random_forest", "xgboost", "svm"]


def _xgboost_classifier() -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )


def make_estimator(name: ModelName) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return _xgboost_classifier()
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model '{name}'.")


@dataclass
class TrainedModel:
    name: ModelName
    pipeline: Pipeline
    feature_names: list[str]
    training_seconds: float
    n_train: int
    n_valid: int
    n_test: int
    class_weight_note: str
    extras: dict[str, Any] = field(default_factory=dict)


def train_model(
    name: ModelName,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
) -> TrainedModel:
    needs_scale = name == "svm"
    preprocessor = build_preprocessor(scale=needs_scale)
    estimator = make_estimator(name)
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    fit_kwargs: dict[str, Any] = {}
    sample_weight = None
    weight_note = "class_weight=balanced on the estimator where supported."
    if name == "xgboost":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kwargs = {"model__sample_weight": sample_weight}
        weight_note = "XGBoost used sklearn compute_sample_weight(class_weight='balanced') on the training fold only."

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    return TrainedModel(
        name=name,
        pipeline=pipe,
        feature_names=list(FEATURE_COLUMNS),
        training_seconds=float(elapsed),
        n_train=int(len(X_train)),
        n_valid=int(len(X_valid) if X_valid is not None else 0),
        n_test=0,
        class_weight_note=weight_note,
        extras={"needs_scale": needs_scale},
    )


def persist_model(trained: TrainedModel, evaluation: dict[str, Any] | None = None) -> None:
    ensure_runtime_dirs()
    payload = {
        "name": trained.name,
        "pipeline": trained.pipeline,
        "feature_names": trained.feature_names,
        "training_seconds": trained.training_seconds,
        "n_train": trained.n_train,
        "n_valid": trained.n_valid,
        "n_test": trained.n_test,
        "class_weight_note": trained.class_weight_note,
        "extras": trained.extras,
    }
    joblib.dump(payload, ACTIVE_MODEL_PATH)
    joblib.dump(trained.pipeline.named_steps["preprocess"], PREPROCESSOR_PATH)
    schema = {
        "feature_names": trained.feature_names,
        "model_name": trained.name,
        "integrity_note": (
            "Metrics stored beside this schema are computed on the last local run. "
            "They are not literature results."
        ),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    if evaluation is not None:
        METRICS_PATH.write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")


def load_persisted_model() -> dict[str, Any] | None:
    if not ACTIVE_MODEL_PATH.exists():
        return None
    return joblib.load(ACTIVE_MODEL_PATH)


def model_status() -> dict[str, Any]:
    ensure_runtime_dirs()
    loaded = load_persisted_model()
    if loaded is None:
        return {"ready": False, "message": "No model has been trained in this project yet."}
    metrics = None
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "ready": True,
        "name": loaded.get("name"),
        "n_train": loaded.get("n_train"),
        "training_seconds": loaded.get("training_seconds"),
        "metrics_available": metrics is not None,
        "metrics_source": metrics.get("source") if metrics else None,
        "path": str(ACTIVE_MODEL_PATH),
    }
