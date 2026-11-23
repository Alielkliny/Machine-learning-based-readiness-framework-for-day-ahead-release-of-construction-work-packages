"""Declared splits. Authority hold-out group by hash."""
import hashlib

SALT = "readiness-2026-11-08"
HOLDOUT_DIGITS = set("cde")


def is_holdout_authority(swa_code) -> bool:
    h = hashlib.sha256(f"{swa_code}|{SALT}".encode()).hexdigest()
    return h[0] in HOLDOUT_DIGITS


SPLITS = {
    "development": ("2023-12-04", "2024-07-31", "dev_authorities"),
    "check_slice": ("2024-08-01", "2024-08-30", "dev_authorities"),
    "R1": ("2024-09-02", "2024-10-31", "all"),
    "R2": ("2024-12-02", "2025-01-24", "all"),
}
HISTORY_START = "2023-10-01"
