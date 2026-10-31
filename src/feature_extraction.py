"""Validate and extract the supported tabular feature schema.

This module does not inspect executables. It only reads numeric columns from
already-extracted feature tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURE_GROUPS, LABEL_COLUMN, METADATA_COLUMNS


@dataclass
class FeatureReport:
    available_features: list[str]
    missing_required: list[str]
    ignored_columns: list[str]
    has_label: bool
    metadata_present: list[str]
    warnings: list[str] = field(default_factory=list)
    is_synthetic_demo: bool = False

    @property
    def can_train(self) -> bool:
        return self.has_label and len(self.missing_required) == 0

    @property
    def can_predict(self) -> bool:
        return len(self.missing_required) == 0


def inspect_columns(columns: Iterable[str]) -> FeatureReport:
    cols = [str(c).strip() for c in columns]
    colset = set(cols)
    available = [c for c in FEATURE_COLUMNS if c in colset]
    missing = [c for c in FEATURE_COLUMNS if c not in colset]
    known = set(FEATURE_COLUMNS) | set(METADATA_COLUMNS) | {LABEL_COLUMN}
    ignored = [c for c in cols if c not in known]
    metadata = [c for c in METADATA_COLUMNS if c in colset]
    warnings: list[str] = []
    if ignored:
        warnings.append(
            "Ignored unsupported columns (not in the project schema): "
            + ", ".join(ignored)
        )
    if missing:
        warnings.append(
            "Missing required feature columns: " + ", ".join(missing)
        )
    is_demo = False
    return FeatureReport(
        available_features=available,
        missing_required=missing,
        ignored_columns=ignored,
        has_label=LABEL_COLUMN in colset,
        metadata_present=metadata,
        warnings=warnings,
        is_synthetic_demo=is_demo,
    )


def extract_feature_frame(df: pd.DataFrame, require_label: bool = False) -> tuple[pd.DataFrame, FeatureReport]:
    report = inspect_columns(df.columns)
    if "data_source" in df.columns:
        sources = {str(v).strip().upper() for v in df["data_source"].dropna().unique()}
        report.is_synthetic_demo = "SYNTHETIC_DEMO" in sources
        if report.is_synthetic_demo:
            report.warnings.append(
                "data_source includes SYNTHETIC_DEMO. Treat metrics as a software "
                "demonstration, not as published experimental results."
            )
    if require_label and not report.has_label:
        raise ValueError("Training requires a 'label' column (benign/ransomware or 0/1).")
    if not report.can_predict:
        raise ValueError(
            "Cannot use this table: missing required features: "
            + ", ".join(report.missing_required)
        )
    keep = list(FEATURE_COLUMNS)
    if report.has_label:
        keep = keep + [LABEL_COLUMN]
    keep_meta = [c for c in METADATA_COLUMNS if c in df.columns]
    out = df.loc[:, keep + keep_meta].copy()
    return out, report


def feature_group_coverage(columns: Iterable[str]) -> dict[str, float]:
    colset = set(columns)
    coverage = {}
    for group, names in FEATURE_GROUPS.items():
        present = sum(1 for n in names if n in colset)
        coverage[group] = present / len(names) if names else 0.0
    return coverage
