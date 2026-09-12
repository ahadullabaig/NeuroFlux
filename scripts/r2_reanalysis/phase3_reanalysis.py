"""
R2 — re-analysing the Phase 3 sweep at the correct unit of analysis.

`docs/phase3_findings.md` was written against a grouping key that omitted
`record_id`, so all 16 MIT-BIH records collapsed into 32 groups and every
reported `n` was 80 where it should have been 5. Between-patient variance was
being counted as within-configuration seed noise — pseudo-replication, and
visible to any reviewer from the *n* alone.

`analysis.fetch_raw_runs` now selects every column and
`compute_statistical_significance` groups on `dataset` and `record_id`, which
turns those 32 groups of 80 into 512 groups of 5. This script re-runs the two
claims that matter under the R2 protocol: seeds averaged within record, paired
Wilcoxon signed-rank across records, Holm-Bonferroni within each family,
bootstrap BCa intervals on the median difference, and a TOST verdict against a
pre-registered margin.

The untrained (`epochs=0`) arm lives in a separate diagnostic database covering
only flicker=50 Hz and tau=20 ms, so the passive-vs-trained comparison is
restricted to that slice of the main sweep. Units pair on (record, input_snr).

Usage:
    uv run python scripts/r2_reanalysis/phase3_reanalysis.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from neuroflux_datasets.analysis import fetch_raw_runs, paired_comparison

SWEEP_DB = Path("results/phase3_full_sweep.db")
UNTRAINED_DB = Path("results/phase3_diagnostics/untrained.db")

# The diagnostic arm was only ever run at these settings.
DIAGNOSTIC_FLICKER_HZ = 50.0
DIAGNOSTIC_TAU_MS = 20.0

# Pre-registered equivalence margins, in each metric's own units. 0.5 dB is the
# roadmap's proposal for SNR; rpeak integrity is a fraction in [0, 1], where a
# 5-point move is already clinically material.
MARGINS = {"delta_snr": 0.5, "rpeak_integrity": 0.05}


def verdict(row: dict[str, Any]) -> str:
    """
    Four outcomes, because significance and equivalence are independent.

    different    — real effect, larger than the margin
    trivial      — real effect, but smaller than the margin
    equivalent   — no effect, and bounded inside the margin
    inconclusive — neither established; underpowered, not invariant
    """
    if row["significant"] and not row["equivalent"]:
        return "different"
    if row["significant"] and row["equivalent"]:
        return "trivial"
    if row["equivalent"]:
        return "equivalent"
    return "inconclusive"


def report(title: str, rows: list[dict[str, Any]]) -> None:
    """Print one family of comparisons."""
    if not rows:
        print(f"\n=== {title} ===\n  no paired units\n")
        return
    print(f"\n=== {title} ===")
    print(
        f"Family size {rows[0]['family_size']}, Holm-Bonferroni corrected. "
        f"Equivalence margin ±{rows[0]['margin']}.\n"
    )
    header = (
        f"{'stratum':>9} {'n':>4} {'median Δ':>10} {'95% CI':>20} "
        f"{'p (Holm)':>10} {'verdict':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        interval = (
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            if math.isfinite(row["ci_low"])
            else "n/a"
        )
        stratum = "all" if row["stratum"] is None else f"{row['stratum']}"
        print(
            f"{stratum:>9} {row['n_pairs']:>4} {row['median_diff']:>+10.3f} "
            f"{interval:>20} {row['p_holm']:>10.4f} {verdict(row):>14}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-db", type=Path, default=SWEEP_DB)
    parser.add_argument("--untrained-db", type=Path, default=UNTRAINED_DB)
    parser.add_argument("--out", type=Path, default=Path("results/phase3_r2"))
    args = parser.parse_args()

    if not args.sweep_db.exists():
        print(f"Sweep database not found: {args.sweep_db}")
        return

    sweep = fetch_raw_runs(args.sweep_db)
    records = {r["record_id"] for r in sweep}
    print(
        f"Loaded {len(sweep)} runs over {len(records)} records from {args.sweep_db}.\n"
        f"The unit of analysis is one configuration per record."
    )

    all_rows: list[dict[str, Any]] = []

    # 1. More training vs less, on the full sweep grid.
    for metric, margin in MARGINS.items():
        rows = paired_comparison(
            sweep,
            factor="epochs",
            level_a=10,
            level_b=50,
            metric=metric,
            stratify_by="input_snr",
            margin=margin,
        )
        report(f"{metric}: 50 epochs minus 10 epochs", rows)
        all_rows.extend(rows)

    # 2. Trained vs passive — the phase3_findings §5.6.2 claim. Restricted to
    #    the slice the diagnostic arm covers.
    if args.untrained_db.exists():
        paired = [
            r
            for r in sweep
            if r["flicker_freq"] == DIAGNOSTIC_FLICKER_HZ
            and r["stdp_tau"] == DIAGNOSTIC_TAU_MS
            and r["epochs"] == 10
        ]
        paired += fetch_raw_runs(args.untrained_db)
        for metric, margin in MARGINS.items():
            rows = paired_comparison(
                paired,
                factor="epochs",
                level_a=0,
                level_b=10,
                metric=metric,
                stratify_by="input_snr",
                margin=margin,
            )
            report(f"{metric}: 10 epochs minus passive (0 epochs)", rows)
            all_rows.extend(rows)
    else:
        print(f"\nUntrained diagnostic not found at {args.untrained_db}; skipping.")

    if not all_rows:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "phase3_reanalysis.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} comparisons to {out_path}")


if __name__ == "__main__":
    main()
