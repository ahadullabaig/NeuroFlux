"""Populate ``mit_bih/manifest.json`` from local provenance sidecars.

Each ``data/mit_bih/<record_id>.provenance.json`` carries the SHA-256
digest of the corresponding ``.dat`` file (computed during download).
This script lifts those digests into the committed manifest so that
:func:`neuroflux_datasets.mit_bih.downloader.download_record` performs a
real integrity check on subsequent cache hits.

Records for which no provenance file exists locally keep their existing
manifest entry (typically ``"dat_checksum": null``).  Run this script
again after downloading additional records to extend coverage.

Usage::

    uv run python scripts/phase3_diagnostics/populate_manifest.py

The script is idempotent and prints a summary of how many records were
updated, skipped (already had a matching checksum), or left null.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "packages"
    / "datasets"
    / "src"
    / "neuroflux_datasets"
    / "mit_bih"
    / "manifest.json"
)
PROVENANCE_DIR = REPO_ROOT / "data" / "mit_bih"


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 1
    if not PROVENANCE_DIR.is_dir():
        print(
            f"ERROR: provenance directory missing at {PROVENANCE_DIR}", file=sys.stderr
        )
        return 1

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    records: dict[str, dict[str, str | None]] = manifest.get("records", {})

    updated: list[str] = []
    confirmed: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    unknown: list[str] = []

    for prov_path in sorted(PROVENANCE_DIR.glob("*.provenance.json")):
        with open(prov_path) as f:
            prov = json.load(f)

        record_id = prov.get("record_id")
        checksum = prov.get("dat_checksum")
        if record_id is None or checksum is None:
            unknown.append(prov_path.name)
            continue

        entry = records.get(record_id)
        if entry is None:
            unknown.append(record_id)
            continue

        existing = entry.get("dat_checksum")
        if existing is None:
            entry["dat_checksum"] = checksum
            updated.append(record_id)
        elif existing == checksum:
            confirmed.append(record_id)
        else:
            mismatched.append((record_id, existing, checksum))

    if mismatched:
        print("ERROR: manifest/provenance mismatch detected:")
        for rid, in_manifest, on_disk in mismatched:
            print(f"  record={rid}: manifest={in_manifest} provenance={on_disk}")
        print("Refusing to overwrite. Investigate before re-running.")
        return 2

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    null_count = sum(
        1 for entry in records.values() if entry.get("dat_checksum") is None
    )

    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    populated = len(records) - null_count
    print(f"  Records with checksums populated: {populated} / {len(records)}")
    print(f"  Newly added in this run:         {len(updated)}")
    print(f"  Already matched (no-op):         {len(confirmed)}")
    print(f"  Still null:                      {null_count}")
    if unknown:
        print(f"  Provenance files without manifest entry: {unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
