"""
Every Phase 3 network against doing nothing, on QRS F1.

`qrs_reanalysis.py` compares epoch budgets with each other. None of those
comparisons says whether any of the networks is better than no processing at
all, and that is the comparison a clinical claim needs. `rpeak_integrity`
could not answer it — it scores the unprocessed signal 1.000 — so it was never
asked properly.

This script scores the **identity baseline** — the noisy signal, untouched —
with the same detector against the same annotations, on exactly the windows
and noise draws `configs/r2/qrs_rescore.yaml` used: `make_pair` is
deterministic in (record, duration, flicker, SNR, seed), so regenerating the
pair reproduces the network's input sample for sample. The identity rows then
enter `paired_comparison` as one more level of the `epochs` factor, and every
network is paired against it on record.

This also corrects `docs/r1_findings.md` §4, whose "passive" arm was built with
`init_scheme="scaled"`, where the Phase 3 runner uses the `Network.feedforward`
default, `"uniform"`. The two passive networks behave very differently on QRS
F1, so R1's regime claim described a network Phase 3 never ran.

Usage:
    uv run python scripts/r2_reanalysis/identity_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_datasets.analysis import fetch_raw_runs, paired_comparison
from neuroflux_datasets.cli import _resolve_reference_beats, _resolve_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_reanalysis import report  # noqa: E402

DEFAULT_DB = Path("results/r2_reanalysis/qrs_rescore.db")
FS = 250.0
IDENTITY = "identity"

# Pre-registered in qrs_reanalysis.py; repeated here, not re-chosen.
F1_MARGIN = 0.05


def identity_rows(runs: list[dict[str, Any]], cache_dir: str) -> list[dict[str, Any]]:
    """One identity row per distinct (record, SNR, seed) in the sweep."""
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for run in runs:
        key = (run["record_id"], run["input_snr"], run["seed"])
        if key in seen:
            continue
        seen.add(key)
        params = {
            "dataset": "mit-bih",
            "record_id": run["record_id"],
            "cache_dir": cache_dir,
            "duration_s": 10.0,
            "flicker_freq": run["flicker_freq"],
            "flicker_amplitude": 0.5,
            "input_snr": run["input_snr"],
            "seed": run["seed"],
        }
        _, noisy = _resolve_signal(params, FS)
        score = qrs_score(_resolve_reference_beats(params, FS), noisy, FS)
        rows.append(
            {
                **{f: run[f] for f in ("dataset", "record_id", "signal_type")},
                "flicker_freq": run["flicker_freq"],
                "input_snr": run["input_snr"],
                "stdp_tau": run["stdp_tau"],
                "seed": run["seed"],
                "epochs": IDENTITY,
                "qrs_f1": score.f1,
                "qrs_sensitivity": score.sensitivity,
                "qrs_ppv": score.ppv,
            }
        )
    return rows


def main() -> None:
    """Score the identity baseline and pair every epoch budget against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache-dir", default="data/mit_bih")
    parser.add_argument(
        "--out", type=Path, default=Path("results/r2_reanalysis/identity_baseline.csv")
    )
    args = parser.parse_args()

    runs = fetch_raw_runs(args.db)
    identity = identity_rows(runs, args.cache_dir)
    print(f"Scored {len(identity)} identity windows.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity[0].keys()))
        writer.writeheader()
        writer.writerows(identity)

    combined = runs + identity
    results: list[dict[str, Any]] = []
    for epochs in (0, 10, 50):
        rows = paired_comparison(
            combined,
            factor="epochs",
            level_a=IDENTITY,
            level_b=epochs,
            metric="qrs_f1",
            stratify_by="input_snr",
            margin=F1_MARGIN,
        )
        label = "passive (0 epochs)" if epochs == 0 else f"{epochs} epochs"
        report(f"qrs_f1: {label} minus identity baseline", rows)
        for row in rows:
            row["family"] = f"{label} minus identity"
        results.extend(rows)

    summary = args.out.with_name("identity_comparisons.csv")
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(identity)} identity rows and {len(results)} comparisons.")


if __name__ == "__main__":
    main()
