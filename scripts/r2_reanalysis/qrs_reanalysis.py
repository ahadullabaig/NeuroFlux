"""
R2 on QRS F1 — re-testing Phase 3's landmark claims with a metric that works.

`phase3_reanalysis.py` applied the R2 protocol to `rpeak_integrity`, and R1 then
showed that metric does not respond to its input at all (`docs/r1_findings.md`
§2). The protocol was sound; the quantity was not. This script repeats the
comparisons on `results/r2_reanalysis/qrs_rescore.db`, a sweep that records the
annotation-referenced QRS scores alongside the old columns, so the two can be
read side by side on identical runs.

Three families, each stratified by input SNR and Holm-corrected within itself:

  10 epochs minus passive   Phase 3 §5.6.2: "STDP destroys R-peak preservation"
  50 epochs minus 10        R2 §3's reversal: more training recovers R-peaks
  50 epochs minus passive   whether any training budget beats none

All three epoch budgets live in one database, generated from one config, so
nothing is merged across files and every pair shares records, windows and
noise draws.

Equivalence margins are fixed below, before the sweep's results were read, and
must not be tuned afterwards. F1, sensitivity and PPV are fractions, so they
take `rpeak_integrity`'s 0.05. Timing jitter takes 10 ms: two and a half
samples at 250 Hz, and well inside the 150 ms EC57 matching window, so a
difference smaller than that is not a clinically distinguishable timing change.

Usage:
    uv run python scripts/r2_reanalysis/qrs_reanalysis.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from neuroflux_datasets.analysis import fetch_raw_runs, paired_comparison

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_reanalysis import report  # noqa: E402

DEFAULT_DB = Path("results/r2_reanalysis/qrs_rescore.db")

MARGINS: dict[str, float] = {
    "delta_snr": 0.5,
    "rpeak_integrity": 0.05,
    "qrs_f1": 0.05,
    "qrs_sensitivity": 0.05,
    "qrs_ppv": 0.05,
    "qrs_jitter_ms": 10.0,
}

FAMILIES: tuple[tuple[str, int, int], ...] = (
    ("10 epochs minus passive (0 epochs)", 0, 10),
    ("50 epochs minus 10 epochs", 10, 50),
    ("50 epochs minus passive (0 epochs)", 0, 50),
)


def main() -> None:
    """Run every family on every metric and write one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--out", type=Path, default=Path("results/r2_reanalysis/qrs_reanalysis.csv")
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}. Run configs/r2/qrs_rescore.yaml first.")
        return

    runs = fetch_raw_runs(args.db)
    records = {r["record_id"] for r in runs}
    seeds = {r["seed"] for r in runs}
    print(
        f"Loaded {len(runs)} runs over {len(records)} records and {len(seeds)} "
        f"seeds from {args.db}.\nThe unit of analysis is one configuration per "
        f"record; seeds are averaged within it."
    )

    all_rows: list[dict[str, Any]] = []
    for title, level_a, level_b in FAMILIES:
        for metric, margin in MARGINS.items():
            rows = paired_comparison(
                runs,
                factor="epochs",
                level_a=level_a,
                level_b=level_b,
                metric=metric,
                stratify_by="input_snr",
                margin=margin,
            )
            report(f"{metric}: {title}", rows)
            for row in rows:
                row["family"] = title
            all_rows.extend(rows)

    if not all_rows:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} comparisons to {args.out}")


if __name__ == "__main__":
    main()
