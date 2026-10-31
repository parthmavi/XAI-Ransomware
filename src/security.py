"""Untrusted-input handling for uploads. Never executes files."""

from __future__ import annotations

from pathlib import Path

from src.config import ALLOWED_UPLOAD_SUFFIXES, MAX_UPLOAD_BYTES


class UploadValidationError(ValueError):
    """Raised when an uploaded file is rejected."""


def validate_upload(filename: str, raw_bytes: bytes) -> str:
    """Return a sanitized basename after checking type, size, and content."""
    if not filename:
        raise UploadValidationError("Empty filename.")
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError(
            f"Rejected file type '{suffix}'. Only CSV feature tables are accepted. "
            "This tool does not scan or execute binaries."
        )
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )
    if len(raw_bytes) == 0:
        raise UploadValidationError("Empty file.")
    if b"\x00" in raw_bytes[:4096]:
        raise UploadValidationError("Binary content detected. CSV text is required.")
    return name
