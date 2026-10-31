"""Shared paths, feature schema, and security limits.

The feature list is a *structured tabular schema* for this project. It is
inspired by categories discussed in the accompanying papers (static PE-like
aggregates and behavioural counts). It is **not** EMBER, CICAndMal2017,
UGRansome, BitcoinHeist, or the Aljabri memory-dump feature set.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
DB_PATH = ARTIFACTS_DIR / "prediction_history.db"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".csv"}
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN = 0.20

LABEL_COLUMN = "label"
BENIGN_LABELS = {"0", "benign", "benignware", "goodware", "false"}
RANSOMWARE_LABELS = {"1", "ransomware", "ransom", "malicious", "true"}

METADATA_COLUMNS = (
    "sample_id",
    "data_source",
    "family",
    "collected_at",
    "notes",
)

# Supported numeric detectors. Unknown extra CSV columns are ignored.
FEATURE_COLUMNS: tuple[str, ...] = (
    "file_entropy",
    "pe_section_count",
    "import_dll_count",
    "suspicious_import_count",
    "packed_section_ratio",
    "debug_directory_present",
    "timestamp_anomaly_score",
    "file_write_count",
    "file_rename_count",
    "file_delete_count",
    "unique_extensions_touched",
    "high_entropy_write_ratio",
    "registry_mod_count",
    "process_create_count",
    "mutex_create_count",
    "dll_load_count",
    "network_connection_count",
    "dns_query_count",
    "bytes_written",
    "encryption_api_like_count",
)

FEATURE_GROUPS = {
    "static_like": (
        "file_entropy",
        "pe_section_count",
        "import_dll_count",
        "suspicious_import_count",
        "packed_section_ratio",
        "debug_directory_present",
        "timestamp_anomaly_score",
    ),
    "behavioural": (
        "file_write_count",
        "file_rename_count",
        "file_delete_count",
        "unique_extensions_touched",
        "high_entropy_write_ratio",
        "registry_mod_count",
        "process_create_count",
        "mutex_create_count",
        "dll_load_count",
        "network_connection_count",
        "dns_query_count",
        "bytes_written",
        "encryption_api_like_count",
    ),
}

FEATURE_HELP = {
    "file_entropy": "Byte-level entropy of the sample representation (0–8).",
    "pe_section_count": "Count of PE-like sections if present; 0 if unknown.",
    "import_dll_count": "Number of imported libraries / modules observed.",
    "suspicious_import_count": "Count of imports often associated with crypto, process, or file APIs.",
    "packed_section_ratio": "Share of sections with packing-like high entropy (0–1).",
    "debug_directory_present": "1 if a debug directory-like flag is set, else 0.",
    "timestamp_anomaly_score": "Score for unusual compile/timestamp patterns (0–1).",
    "file_write_count": "File write operations in the observation window.",
    "file_rename_count": "File rename operations in the observation window.",
    "file_delete_count": "File delete operations in the observation window.",
    "unique_extensions_touched": "Distinct file extensions written or renamed.",
    "high_entropy_write_ratio": "Share of writes with high-entropy content (0–1).",
    "registry_mod_count": "Registry create/set/delete operations.",
    "process_create_count": "New processes spawned.",
    "mutex_create_count": "Mutex objects created.",
    "dll_load_count": "Dynamic libraries loaded.",
    "network_connection_count": "Outbound connections observed.",
    "dns_query_count": "DNS queries observed.",
    "bytes_written": "Total bytes written to disk in the window.",
    "encryption_api_like_count": "Calls resembling cryptographic APIs (count, not proof of encryption).",
}

CLASS_NAMES = {0: "benign", 1: "ransomware"}
SUPPORTED_MODELS = ("random_forest", "xgboost", "svm")
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "last_evaluation.json"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

DEMO_BANNER = (
    "SYNTHETIC_DEMO: rows in the bundled sample CSV are generated for software "
    "demonstration. They are not captured ransomware telemetry and must not be "
    "cited as experimental results."
)


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, MODELS_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
