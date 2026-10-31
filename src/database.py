"""Optional SQLite store for prediction history. No credentials are stored."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import DB_PATH, ensure_runtime_dirs


def _connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_name TEXT,
            predicted_class TEXT,
            p_ransomware REAL,
            source_name TEXT,
            is_synthetic INTEGER,
            row_payload TEXT,
            explanation_summary TEXT
        )
        """
    )
    conn.commit()
    return conn


def record_prediction(
    *,
    model_name: str,
    predicted_class: str,
    p_ransomware: float | None,
    source_name: str,
    is_synthetic: bool,
    row_payload: dict[str, Any],
    explanation_summary: str,
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO predictions (
                created_at, model_name, predicted_class, p_ransomware,
                source_name, is_synthetic, row_payload, explanation_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model_name,
                predicted_class,
                p_ransomware,
                source_name,
                1 if is_synthetic else 0,
                json.dumps(row_payload, default=str),
                explanation_summary,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_history(limit: int = 200) -> pd.DataFrame:
    conn = _connect()
    try:
        frame = pd.read_sql_query(
            "SELECT id, created_at, model_name, predicted_class, p_ransomware, "
            "source_name, is_synthetic, explanation_summary FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
    except Exception:
        frame = pd.DataFrame()
    finally:
        conn.close()
    return frame


def history_counts() -> dict[str, int]:
    conn = _connect()
    try:
        cur = conn.execute("SELECT COUNT(*), SUM(CASE WHEN predicted_class = 'ransomware' THEN 1 ELSE 0 END) FROM predictions")
        total, ransom = cur.fetchone()
        return {"total": int(total or 0), "ransomware_alerts": int(ransom or 0)}
    finally:
        conn.close()
