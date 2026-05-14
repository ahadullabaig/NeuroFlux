"""MIT-BIH Arrhythmia Database downloader with SHA-256 integrity verification.

Downloads individual records from PhysioNet via the ``wfdb`` library, caches
them locally under a configurable directory (default ``data/mit_bih/``), and
verifies file integrity against a committed SHA-256 manifest.

The ``wfdb`` import is **lazy** — it only happens inside functions that
actually hit the network — so that Phase 2 code paths never depend on it.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MITDB_NAME = "mitdb"
_PHYSIONET_URL = "https://physionet.org/content/mitdb/1.0.0/"
_MANIFEST_PATH = Path(__file__).parent / "manifest.json"


class IntegrityError(Exception):
    """Raised when a cached file fails SHA-256 verification."""


# ---------------------------------------------------------------------------
# Checksum utilities
# ---------------------------------------------------------------------------


def compute_checksum(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Lowercase hex-encoded SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_manifest() -> dict[str, Any]:
    """Load the SHA-256 manifest from the committed JSON file.

    Returns:
        Parsed manifest dict with a ``records`` key mapping record IDs
        to ``{"dat_checksum": ..., "source_url": ...}`` entries.
    """
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH) as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {"database": _MITDB_NAME, "records": {}}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _write_provenance(record_id: str, cache_dir: Path) -> None:
    """Persist download metadata alongside the cached record files.

    Creates ``<record_id>.provenance.json`` in *cache_dir* recording the
    source URL, download timestamp, and computed SHA-256 of the ``.dat``
    file (if present).
    """
    dat_path = cache_dir / f"{record_id}.dat"
    dat_checksum: str | None = None
    if dat_path.exists():
        dat_checksum = compute_checksum(dat_path)

    provenance: dict[str, Any] = {
        "record_id": record_id,
        "source_url": f"{_PHYSIONET_URL}{record_id}",
        "download_timestamp": datetime.now(UTC).isoformat(),
        "database": _MITDB_NAME,
        "dat_checksum": dat_checksum,
    }
    provenance_path = cache_dir / f"{record_id}.provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_record(record_id: str, cache_dir: str | Path) -> Path:
    """Download a single MIT-BIH record, caching locally.

    On a **cache hit** the ``.dat`` file's SHA-256 is checked against the
    committed manifest (if a checksum is present). A mismatch deletes the
    stale cache and re-downloads.

    On a **fresh download** the checksum is verified post-download (if the
    manifest entry exists) and provenance metadata is written to a
    ``<record_id>.provenance.json`` sidecar file.

    Args:
        record_id: MIT-BIH record identifier (e.g. ``"100"``, ``"234"``).
        cache_dir: Local directory to cache downloaded files.

    Returns:
        Resolved *cache_dir* :class:`Path` (the directory containing
        ``<record_id>.hea``, ``<record_id>.dat``, etc.).

    Raises:
        IntegrityError: If the downloaded ``.dat`` file does not match
            the expected SHA-256 from the manifest.
    """
    import wfdb  # type: ignore[import-untyped]

    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    record_hea = cache_path / f"{record_id}.hea"
    record_dat = cache_path / f"{record_id}.dat"

    manifest = _load_manifest()
    record_manifest = manifest.get("records", {}).get(record_id, {})
    expected_checksum: str | None = record_manifest.get("dat_checksum")

    # --- Cache hit path ---
    if record_hea.exists() and record_dat.exists():
        if expected_checksum is not None:
            actual = compute_checksum(record_dat)
            if actual == expected_checksum:
                return cache_path
            # Mismatch — remove stale files and re-download below
            record_hea.unlink(missing_ok=True)
            record_dat.unlink(missing_ok=True)
            atr = cache_path / f"{record_id}.atr"
            atr.unlink(missing_ok=True)
        else:
            # No manifest checksum → trust cache
            return cache_path

    # --- Download ---
    wfdb.dl_database(
        _MITDB_NAME,
        str(cache_path),
        records=[record_id],
    )

    # --- Post-download integrity check ---
    if expected_checksum is not None and record_dat.exists():
        actual = compute_checksum(record_dat)
        if actual != expected_checksum:
            raise IntegrityError(
                f"Record {record_id}: SHA-256 mismatch after download. "
                f"Expected {expected_checksum}, got {actual}."
            )

    # --- Provenance sidecar ---
    _write_provenance(record_id, cache_path)

    return cache_path


def verify_records(
    record_ids: list[str],
    cache_dir: str | Path,
) -> list[str]:
    """Check which records are missing from the local cache.

    This is a **header-only** check — it only looks for the existence of
    ``.hea`` and ``.dat`` files, with no network access.  Used by the
    ``--dry-run`` CLI flag.

    Args:
        record_ids: Record identifiers to check.
        cache_dir: Local cache directory.

    Returns:
        List of record IDs whose files are **not** present in *cache_dir*.
    """
    cache_path = Path(cache_dir)
    missing: list[str] = []
    for rid in record_ids:
        hea = cache_path / f"{rid}.hea"
        dat = cache_path / f"{rid}.dat"
        if not (hea.exists() and dat.exists()):
            missing.append(rid)
    return missing
