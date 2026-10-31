"""Create a clearly labelled synthetic demonstration CSV.

The rows are generated from overlapping class-conditional distributions.
They are not captured from ransomware or benign software.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, FEATURE_COLUMNS, RANDOM_STATE


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate(n_benign: int = 220, n_ransomware: int = 180, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def draw(label: int, n: int, family: str) -> None:
        ransom = label == 1
        entropy_mean = 6.4 if ransom else 5.1
        write_mean = 420 if ransom else 80
        rename_mean = 90 if ransom else 6
        enc_ratio_mean = 0.62 if ransom else 0.12
        for i in range(n):
            row = {
                "sample_id": f"{'R' if ransom else 'B'}-{family}-{i:04d}",
                "data_source": "SYNTHETIC_DEMO",
                "family": family,
                "collected_at": "not_a_capture",
                "notes": "synthetic demonstration row; not telemetry",
                "file_entropy": float(_clip(rng.normal(entropy_mean, 0.55), 0.5, 8.0)),
                "pe_section_count": int(_clip(rng.normal(7 if ransom else 5, 2), 1, 16)),
                "import_dll_count": int(_clip(rng.normal(18 if ransom else 14, 6), 1, 80)),
                "suspicious_import_count": int(_clip(rng.normal(9 if ransom else 3, 3), 0, 40)),
                "packed_section_ratio": float(_clip(rng.normal(0.45 if ransom else 0.12, 0.18), 0, 1)),
                "debug_directory_present": int(rng.random() < (0.15 if ransom else 0.45)),
                "timestamp_anomaly_score": float(_clip(rng.normal(0.55 if ransom else 0.22, 0.2), 0, 1)),
                "file_write_count": int(_clip(rng.normal(write_mean, 90 if ransom else 40), 0, 2500)),
                "file_rename_count": int(_clip(rng.normal(rename_mean, 25 if ransom else 4), 0, 400)),
                "file_delete_count": int(_clip(rng.normal(40 if ransom else 8, 15), 0, 300)),
                "unique_extensions_touched": int(_clip(rng.normal(18 if ransom else 5, 6), 1, 60)),
                "high_entropy_write_ratio": float(_clip(rng.normal(enc_ratio_mean, 0.15), 0, 1)),
                "registry_mod_count": int(_clip(rng.normal(35 if ransom else 10, 12), 0, 200)),
                "process_create_count": int(_clip(rng.normal(8 if ransom else 3, 3), 0, 40)),
                "mutex_create_count": int(_clip(rng.normal(6 if ransom else 2, 2), 0, 30)),
                "dll_load_count": int(_clip(rng.normal(22 if ransom else 12, 7), 1, 80)),
                "network_connection_count": int(_clip(rng.normal(12 if ransom else 8, 6), 0, 80)),
                "dns_query_count": int(_clip(rng.normal(9 if ransom else 7, 5), 0, 60)),
                "bytes_written": float(_clip(rng.normal(8.5e6 if ransom else 1.2e6, 2.5e6), 0, 5e7)),
                "encryption_api_like_count": int(_clip(rng.normal(14 if ransom else 2, 5), 0, 80)),
                "label": "ransomware" if ransom else "benign",
            }
            rows.append(row)

    draw(0, n_benign // 2, "demo_office_suite")
    draw(0, n_benign - n_benign // 2, "demo_browser")
    draw(1, n_ransomware // 3, "demo_family_alpha")
    draw(1, n_ransomware // 3, "demo_family_beta")
    draw(1, n_ransomware - 2 * (n_ransomware // 3), "demo_family_gamma")
    frame = pd.DataFrame(rows)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise RuntimeError(f"generator missing columns: {missing}")
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "sample_synthetic_demo.csv"
    frame = generate()
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} SYNTHETIC_DEMO rows to {path}")


if __name__ == "__main__":
    main()
