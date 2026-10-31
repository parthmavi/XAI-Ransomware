"""Cleaning, encoding, leakage-aware splitting, and train-only transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BENIGN_LABELS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RANDOM_STATE,
    RANSOMWARE_LABELS,
    TEST_SIZE,
    VALID_SIZE_WITHIN_TRAIN,
)


class LabelEncodingError(ValueError):
    pass


def encode_labels(series: pd.Series) -> pd.Series:
    mapped: list[int] = []
    for raw in series.tolist():
        if pd.isna(raw):
            raise LabelEncodingError("Missing label values are not allowed.")
        key = str(raw).strip().lower()
        if key in BENIGN_LABELS:
            mapped.append(0)
        elif key in RANSOMWARE_LABELS:
            mapped.append(1)
        else:
            raise LabelEncodingError(
                f"Unrecognised label '{raw}'. Use 0/1 or benign/ransomware."
            )
    return pd.Series(mapped, index=series.index, name=LABEL_COLUMN, dtype=int)


def coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    subset = [c for c in FEATURE_COLUMNS if c in df.columns]
    if LABEL_COLUMN in df.columns:
        subset = subset + [LABEL_COLUMN]
    cleaned = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return cleaned, before - len(cleaned)


def class_balance(y: pd.Series) -> dict[str, Any]:
    counts = y.value_counts().to_dict()
    total = int(len(y))
    return {
        "n_samples": total,
        "n_benign": int(counts.get(0, 0)),
        "n_ransomware": int(counts.get(1, 0)),
        "ransomware_share": float(counts.get(1, 0) / total) if total else 0.0,
    }


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    dropped_duplicates: int
    notes: list[str]


def leakage_aware_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE_WITHIN_TRAIN,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Stratified random split after exact-duplicate removal.

    Family-disjoint and temporal splits are documented as future work: the
    bundled schema has optional `family` / `collected_at` metadata but the
    synthetic demo is not a chronological capture.
    """
    notes: list[str] = []
    work = coerce_numeric_features(df)
    work, dropped = drop_exact_duplicates(work)
    if dropped:
        notes.append(f"Removed {dropped} exact duplicate feature+label rows.")
    if LABEL_COLUMN not in work.columns:
        raise ValueError("label column required for splitting.")
    y = encode_labels(work[LABEL_COLUMN])
    X = work.loc[:, list(FEATURE_COLUMNS)]
    if y.nunique() < 2:
        raise ValueError("Need both benign and ransomware labels to train.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_size,
        random_state=random_state,
        stratify=y_train_full,
    )
    notes.append(
        "Split is stratified and random. It is not family-disjoint or temporal. "
        "Do not treat test metrics as zero-day generalisation."
    )
    return SplitResult(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        dropped_duplicates=dropped,
        notes=notes,
    )


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Impute with train-fitted medians. Scale only when the model needs it (SVM)."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=steps)
    return ColumnTransformer(
        transformers=[("num", numeric, list(FEATURE_COLUMNS))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    arr = preprocessor.transform(X.loc[:, list(FEATURE_COLUMNS)])
    names = list(FEATURE_COLUMNS)
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            names = list(FEATURE_COLUMNS)
    return pd.DataFrame(np.asarray(arr), columns=names, index=getattr(X, "index", None))
