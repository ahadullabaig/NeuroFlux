"""
R5 analysis — the pre-registered tests for the encoder x decoder grid (R5a) and
the QRS detector check (R5b).

Each rule is stated in the docstring of the grid script that produced the data
(`encoders_decoders.py`, `qrs_detectors.py`), written before that grid ran. The
statistics helpers are R3's (`analyse_r3.py`): records are units on MIT-BIH and
seeds elsewhere, SNR averages `ok` runs and F1 every run, paired tests pair
units, Wilcoxon + Holm + BCa + TOST.

Usage:
    uv run python scripts/r5_generality/analyse_r5.py
    uv run python scripts/r5_generality/analyse_r5.py --part detectors
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from neuroflux_datasets.analysis import bootstrap_median_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
import analyse_r3 as r3  # noqa: E402
from r3_common import split_floats  # noqa: E402

RESULTS = Path("results/r5_generality")
r3.OUT = RESULTS / "analysis"

PRIMARY = "snr_best_lag_db"
CAUSAL = "snr_affine_db"
SIGNALS: tuple[str, ...] = ("mit-bih", "synthetic-ecg", "eeg", "sine")
ENCODERS: tuple[str, ...] = ("poisson", "temporal", "delta")
RATES: tuple[str, ...] = ("40", "5")
ESCAPE_DB = 3.0
SLOPE_RANGE = (10.0, 40.0)
# (spike-rate window, population tau) with the same noise bandwidth.
MATCHED_PAIRS: tuple[tuple[str, str], ...] = (
    ("spike-rate-20", "population-10"),
    ("spike-rate-40", "population-20"),
    ("spike-rate-80", "population-40"),
)
CLEAN_LEVEL = math.inf

Row = dict[str, str]
Record = dict[str, Any]


def source(name: str) -> Path | None:
    """The plain CSV a grid wrote, or its committed gzipped copy."""
    plain = RESULTS / name
    packed = plain.with_name(plain.name + ".gz")
    if plain.exists():
        return plain
    return packed if packed.exists() else None


def readout_label(row: Row) -> str:
    if row["decoder"] == "latency":
        return "latency"
    return f"{row['decoder']}-{int(float(row['decoder_tau']))}"


def output_rate(row: Row) -> str:
    return f"{split_floats(row['layer_rate_targets_hz'])[-1]:g}"


# --------------------------------------------------------------------------
# R5a — encoders x decoders
# --------------------------------------------------------------------------


def check_against_c1(rows: list[Row]) -> None:
    """Poisson, population tau = 20, 40 Hz must equal C1's controlled arm."""
    c1_path = r3.raw("c1_flatness.csv")
    stored = {
        (r["signal"], r["unit"], r["seed"], r["input_snr_nominal"]): r
        for r in r3.load(c1_path)
        if r["operating_point"] == "controlled"
        and r["epochs"] == "0"
        and r["interference"] == "gaussian"
    }
    checked = 0
    for row in rows:
        if not (
            row["encoder"] == "poisson"
            and readout_label(row) == "population-20"
            and output_rate(row) == "40"
        ):
            continue
        key = (row["signal"], row["unit"], row["seed"], row["input_snr_nominal"])
        reference = stored[key]
        for field in ("snr_affine_db", "stage_affine_db", "verdict", "out_rate_hz"):
            if row[field] != reference[field]:
                raise SystemExit(
                    f"R5a does not reproduce C1 at {key}: {field} "
                    f"{row[field]} != {reference[field]}"
                )
        checked += 1
    print(f"  Check: {checked} Poisson runs reproduce C1's controlled arm exactly.")


def cells(
    rows: list[Row],
) -> dict[tuple[str, str, str, str], dict[float, list[Row]]]:
    """(signal, rate, encoder, readout) -> input level -> rows."""
    grouped: defaultdict[tuple[str, str, str, str], defaultdict[float, list[Row]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        key = (row["signal"], output_rate(row), row["encoder"], readout_label(row))
        grouped[key][r3.fnum(row["input_snr_nominal"])].append(row)
    return {k: dict(v) for k, v in grouped.items()}


def best_readouts(
    grid: dict[tuple[str, str, str, str], dict[float, list[Row]]],
) -> dict[tuple[str, str, str], str]:
    """The readout with the highest median noiseless ceiling on the primary metric."""
    best: dict[tuple[str, str, str], tuple[float, str]] = {}
    for (signal, rate, encoder, readout), levels in sorted(grid.items()):
        ceiling = r3.median(primary_values(levels[CLEAN_LEVEL]).values())
        current = best.get((signal, rate, encoder))
        if math.isfinite(ceiling) and (current is None or ceiling > current[0]):
            best[(signal, rate, encoder)] = (ceiling, readout)
    return {key: readout for key, (_, readout) in best.items()}


def primary_values(rows: Iterable[Row]) -> dict[str, float]:
    """
    Compensated SNR per unit over `ok` runs.

    `best_lag_snr` returns NaN for a live output that has no positive-scale fit
    at any lag — it is anti-correlated with the signal throughout. A fit whose
    scale is constrained to be non-negative puts that scale at zero, which
    scores exactly 0 dB, so that is the value such a run counts as.
    """
    filled = []
    for row in rows:
        if row["verdict"] == "ok" and not math.isfinite(r3.fnum(row[PRIMARY])):
            row = {**row, PRIMARY: "0.0"}
        filled.append(row)
    return r3.unit_values(filled, PRIMARY)


def ok_share(rows: Iterable[Row]) -> float:
    verdicts = [r["verdict"] == "ok" for r in rows]
    return float(np.mean(verdicts)) if verdicts else math.nan


def analyse_encoders() -> None:
    path = source("encoders_decoders.csv")
    if path is None:
        print("encoders_decoders.csv not found")
        return
    rows = r3.load(path)
    print(f"\nR5a — encoders x decoders ({len(rows)} rows)")
    check_against_c1(rows)
    grid = cells(rows)
    best = best_readouts(grid)

    # Every cell, for Table II and the decoder comparison.
    table: list[Record] = []
    for (signal, rate, encoder, readout), levels in sorted(grid.items()):
        record: Record = {
            "signal": signal,
            "output_rate_hz": rate,
            "encoder": encoder,
            "readout": readout,
            "best": best.get((signal, rate, encoder)) == readout,
            "ok_share": ok_share(r for group in levels.values() for r in group),
        }
        for level in sorted(levels):
            label = "clean" if math.isinf(level) else f"{level:g}dB"
            record[f"compensated_{label}"] = r3.median(
                primary_values(levels[level]).values()
            )
            record[f"causal_{label}"] = r3.median(
                r3.unit_values(levels[level], CAUSAL).values()
            )
        lags = r3.unit_values(levels[CLEAN_LEVEL], "best_lag_ms")
        record["best_lag_ms_clean"] = r3.median(lags.values())
        stages = [
            split_floats(r["stage_affine_db"])
            for r in levels[CLEAN_LEVEL]
            if r["verdict"] == "ok"
        ]
        if stages:
            per_stage = np.nanmedian(np.array(stages), axis=0)
            record["causal_stages_clean"] = ";".join(f"{v:.2f}" for v in per_stage)
        table.append(record)
    r3.write("r5_encoder_cells.csv", table)

    print("\n  Best readout and noiseless ceiling (median over units, dB):")
    print(
        f"  {'signal':14} {'rate':>4} {'encoder':9} {'readout':15} "
        f"{'compensated':>11} {'causal':>7} {'ok':>5}"
    )
    for record in table:
        if record["best"]:
            print(
                f"  {record['signal']:14} {record['output_rate_hz']:>4} "
                f"{record['encoder']:9} {record['readout']:15} "
                f"{record['compensated_clean']:>11.2f} "
                f"{record['causal_clean']:>7.2f} {record['ok_share']:>5.2f}"
            )

    def ceiling_values(signal: str, rate: str, encoder: str) -> dict[str, float]:
        readout = best[(signal, rate, encoder)]
        return primary_values(grid[(signal, rate, encoder, readout)][CLEAN_LEVEL])

    # T1 — does the ceiling disappear under temporal coding?
    t1: list[Record] = []
    for rate in RATES:
        for other in ("temporal", "delta"):
            family: list[Record] = []
            for signal in SIGNALS:
                result = r3.compare(
                    ceiling_values(signal, rate, "poisson"),
                    ceiling_values(signal, rate, other),
                    r3.SNR_MARGIN,
                )
                result.update(
                    {
                        "output_rate_hz": rate,
                        "encoder": other,
                        "signal": signal,
                        "poisson_readout": best[(signal, rate, "poisson")],
                        "encoder_readout": best[(signal, rate, other)],
                    }
                )
                family.append(result)
            r3.holm(family)
            for result in family:
                result["escapes"] = (
                    result["significant"] and result["median_diff"] >= ESCAPE_DB
                )
            t1.extend(family)
    r3.write("r5_t1_encoders.csv", t1)
    print(
        "\n  T1 — encoder minus Poisson noiseless ceiling (compensated, best readouts):"
    )
    for result in t1:
        print(
            f"  {result['output_rate_hz']:>2} Hz {result['encoder']:8} "
            f"{result['signal']:14} {r3.fmt(result['median_diff'])} "
            f"[{r3.fmt(result['ci_low'])}, {r3.fmt(result['ci_high'])}] "
            f"p_holm {result['p_holm']:.2g} equiv {result['equivalent']!s:5} "
            f"escapes {result['escapes']}"
        )
    decisive = [
        r for r in t1 if r["output_rate_hz"] == "40" and r["encoder"] == "temporal"
    ]
    escapes = sum(bool(r["escapes"]) for r in decisive)
    print(
        f"  T1 decision: temporal coding escapes on {escapes} of {len(decisive)} "
        f"signals -> the ceiling is {'' if escapes >= 3 else 'NOT '}"
        "specific to rate coding"
    )

    # T2 — flatness per encoder.
    t2: list[Record] = []
    for (signal, rate, encoder), readout in sorted(best.items()):
        curve = {
            level: primary_values(group)
            for level, group in grid[(signal, rate, encoder, readout)].items()
        }
        slopes = r3.unit_slopes(curve, *SLOPE_RANGE)
        values = [slopes[u] for u in sorted(slopes)]
        low, high = bootstrap_median_ci(values, seed=0)
        t2.append(
            {
                "signal": signal,
                "output_rate_hz": rate,
                "encoder": encoder,
                "readout": readout,
                "n_units": len(values),
                "slope_median": r3.median(values),
                "ci_low": low,
                "ci_high": high,
                "flat": math.isfinite(high) and high < 1.0,
            }
        )
    r3.write("r5_t2_flatness.csv", t2)
    print(
        "\n  T2 — slope of output on input SNR, 10-40 dB (compensated, best readout):"
    )
    for result in t2:
        print(
            f"  {result['signal']:14} {result['output_rate_hz']:>2} Hz "
            f"{result['encoder']:9} {result['slope_median']:+.3f} "
            f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] flat {result['flat']}"
        )

    # T3 — decoder type at matched bandwidth (Poisson).
    t3: list[Record] = []
    for signal in SIGNALS:
        for rate in RATES:
            pairs = list(MATCHED_PAIRS)
            spike_best = max(
                (r for r in ("spike-rate-20", "spike-rate-40", "spike-rate-80")),
                key=lambda r: r3.median(
                    primary_values(
                        grid[(signal, rate, "poisson", r)][CLEAN_LEVEL]
                    ).values()
                ),
            )
            population_best = max(
                ("population-10", "population-20", "population-40"),
                key=lambda r: r3.median(
                    primary_values(
                        grid[(signal, rate, "poisson", r)][CLEAN_LEVEL]
                    ).values()
                ),
            )
            pairs.append((spike_best, population_best))
            for index, (spike, population) in enumerate(pairs):
                result = r3.compare(
                    primary_values(
                        grid[(signal, rate, "poisson", population)][CLEAN_LEVEL]
                    ),
                    primary_values(grid[(signal, rate, "poisson", spike)][CLEAN_LEVEL]),
                    r3.SNR_MARGIN,
                )
                result.update(
                    {
                        "signal": signal,
                        "output_rate_hz": rate,
                        "comparison": "best" if index == len(pairs) - 1 else "matched",
                        "spike_rate": spike,
                        "population": population,
                    }
                )
                t3.append(result)
    r3.holm(t3)
    for result in t3:
        result["spike_rate_higher"] = (
            result["significant"] and result["median_diff"] > r3.SNR_MARGIN
        )
    r3.write("r5_t3_decoders.csv", t3)
    print(
        "\n  T3 — spike-rate minus population, noiseless, compensated (Holm over all):"
    )
    for result in t3:
        print(
            f"  {result['signal']:14} {result['output_rate_hz']:>2} Hz "
            f"{result['spike_rate']:13} - {result['population']:13} "
            f"{r3.fmt(result['median_diff'])} "
            f"[{r3.fmt(result['ci_low'])}, {r3.fmt(result['ci_high'])}] "
            f"p_holm {result['p_holm']:.2g} equiv {result['equivalent']!s:5}"
        )
    higher = sum(bool(r["spike_rate_higher"]) for r in t3)
    print(
        f"  T3 decision: {higher} of {len(t3)} comparisons have spike-rate "
        f"significantly higher by > 0.5 dB -> 'not a population-decoder "
        f"artefact' {'NOT established' if higher else 'established'}"
    )
    print(
        "  (Holm across all 32: with 10 seeds the smallest exact Wilcoxon p is "
        "0.002, so no synthetic-signal comparison can reach significance.)"
    )

    # T4 — delta at a calibrated threshold.
    delta_rows = [r for r in rows if r["encoder"] == "delta"]
    by_run: defaultdict[str, list[Row]] = defaultdict(list)
    for row in delta_rows:
        by_run[row["spec_key"]].append(row)
    run_ok = float(
        np.mean([all(r["verdict"] == "ok" for r in g) for g in by_run.values()])
    )
    share = ok_share(delta_rows)
    print(
        f"\n  T4 — delta: {share:.3f} of rows ok, {run_ok:.3f} of runs ok under "
        f"every readout -> {'met' if run_ok >= 0.95 else 'NOT met'} (>= 0.95)"
    )
    thresholds = [r3.fnum(r["delta_threshold"]) for r in delta_rows]
    rates = [r3.fnum(r["encoder_rate_hz"]) for r in delta_rows]
    print(
        f"  delta thresholds {np.nanmin(thresholds):.3g} to "
        f"{np.nanmax(thresholds):.3g}; encoder rates {min(rates):.1f}-"
        f"{max(rates):.1f} Hz"
    )


# --------------------------------------------------------------------------
# R5b — QRS detectors
# --------------------------------------------------------------------------

DETECTORS: tuple[str, ...] = ("neurokit", "pantompkins1985", "hamilton2002")
CLEANINGS: tuple[str, ...] = ("raw", "ecg_clean")


def condition(row: Row) -> str:
    if row["experiment"] == "c3-rate":
        return f"rate-{output_rate(row)}Hz"
    return f"shipped-{'trained' if row['epochs'] != '0' else 'passive'}"


def record_f1(rows: Iterable[Row]) -> dict[str, float]:
    """Mean F1 per record over every run."""
    buckets: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[row["unit"]].append(float(row["f1"]))
    return {unit: float(np.mean(values)) for unit, values in sorted(buckets.items())}


def analyse_detectors() -> None:
    path = source("qrs_detectors.csv")
    if path is None:
        print("qrs_detectors.csv not found")
        return
    rows = r3.load(path)
    print(f"\nR5b — QRS detectors ({len(rows)} rows)")

    index: defaultdict[tuple[str, float, str, str, str], list[Row]] = defaultdict(list)
    inputs: defaultdict[tuple[float, str, str], dict[tuple[str, str], Row]] = (
        defaultdict(dict)
    )
    for row in rows:
        level = r3.fnum(row["input_snr_db"])
        if row["source"] == "output":
            index[
                (condition(row), level, row["detector"], row["cleaning"], "output")
            ].append(row)
        else:
            # The same noisy window appears in several conditions; keep one copy.
            inputs[(level, row["detector"], row["cleaning"])][
                (row["unit"], row["seed"])
            ] = row

    def output_f1(
        name: str, level: float, detector: str, cleaning: str
    ) -> dict[str, float]:
        return record_f1(index.get((name, level, detector, cleaning, "output"), []))

    def input_f1(level: float, detector: str, cleaning: str) -> dict[str, float]:
        return record_f1(inputs.get((level, detector, cleaning), {}).values())

    # Descriptive: every condition under every variant.
    levels = sorted({key[1] for key in index})
    conditions = sorted({key[0] for key in index})
    table: list[Record] = []
    for level in levels:
        for detector in DETECTORS:
            for cleaning in CLEANINGS:
                base = input_f1(level, detector, cleaning)
                if base:
                    table.append(
                        {
                            "input_snr_db": level,
                            "condition": "input",
                            "detector": detector,
                            "cleaning": cleaning,
                            "f1_median": r3.median(base.values()),
                        }
                    )
                for name in conditions:
                    values = output_f1(name, level, detector, cleaning)
                    if values:
                        table.append(
                            {
                                "input_snr_db": level,
                                "condition": name,
                                "detector": detector,
                                "cleaning": cleaning,
                                "f1_median": r3.median(values.values()),
                            }
                        )
    r3.write("r5_detector_cells.csv", table)

    # Q1 — sparse output.
    q1: list[Record] = []
    for detector in DETECTORS:
        for cleaning in CLEANINGS:
            result = r3.compare(
                output_f1("rate-40Hz", 10.0, detector, cleaning),
                output_f1("rate-1Hz", 10.0, detector, cleaning),
                r3.F1_MARGIN,
            )
            result.update({"detector": detector, "cleaning": cleaning})
            q1.append(result)
    r3.holm(q1)
    print("\n  Q1 — F1 at 10 dB, output 1 Hz minus 40 Hz:")
    for result in q1:
        result["survives"] = result["significant"] and result["median_diff"] > 0
        print(
            f"  {result['detector']:16} {result['cleaning']:9} "
            f"40 Hz {result['reference_median']:.2f} 1 Hz {result['level_median']:.2f} "
            f"diff {r3.fmt(result['median_diff'])} "
            f"[{r3.fmt(result['ci_low'])}, {r3.fmt(result['ci_high'])}] "
            f"p_holm {result['p_holm']:.2g} survives {result['survives']}"
        )
    survived = sum(bool(r["survives"]) for r in q1)
    print(f"  Q1 decision: survives {survived} of 6 variants")
    r3.write("r5_q1_sparse_output.csv", q1)

    # Q2 — training over passive, shipped.
    q2: list[Record] = []
    shipped_levels = sorted({key[1] for key in index if key[0].startswith("shipped")})
    for detector in DETECTORS:
        for cleaning in CLEANINGS:
            for level in shipped_levels:
                result = r3.compare(
                    output_f1("shipped-passive", level, detector, cleaning),
                    output_f1("shipped-trained", level, detector, cleaning),
                    r3.F1_MARGIN,
                )
                result.update(
                    {"detector": detector, "cleaning": cleaning, "input_snr_db": level}
                )
                q2.append(result)
    r3.holm(q2)
    print("\n  Q2 — shipped network F1, 10 epochs minus passive:")
    variants_surviving = 0
    for detector in DETECTORS:
        for cleaning in CLEANINGS:
            group = [
                r for r in q2 if r["detector"] == detector and r["cleaning"] == cleaning
            ]
            survives = all(r["significant"] and r["median_diff"] > 0 for r in group)
            variants_surviving += survives
            cells_text = "  ".join(
                f"{r['input_snr_db']:+g}:{r3.fmt(r['median_diff'])}"
                f"{'*' if r['significant'] else ''}"
                for r in group
            )
            print(f"  {detector:16} {cleaning:9} {cells_text}  survives {survives}")
    print(
        f"  Q2 decision: survives {variants_surviving} of 6 variants "
        "(* = p_holm < 0.05)"
    )
    r3.write("r5_q2_training.csv", q2)

    # Q3 — the cleaned input against the best network condition.
    q3: list[Record] = []
    for level in levels:
        for detector in DETECTORS:
            reference = input_f1(level, detector, "ecg_clean")
            if not reference:
                continue
            candidates = [
                (name, cleaning)
                for name in conditions
                for cleaning in CLEANINGS
                if output_f1(name, level, detector, cleaning)
            ]
            name, cleaning = max(
                candidates,
                key=lambda c: r3.median(
                    output_f1(c[0], level, detector, c[1]).values()
                ),
            )
            result = r3.compare(
                reference, output_f1(name, level, detector, cleaning), r3.F1_MARGIN
            )
            result.update(
                {
                    "input_snr_db": level,
                    "detector": detector,
                    "best_condition": name,
                    "best_cleaning": cleaning,
                }
            )
            q3.append(result)
    r3.holm(q3)
    print("\n  Q3 — best network condition minus the cleaned noisy input:")
    network_wins = 0
    for result in q3:
        wins = result["significant"] and result["median_diff"] > 0
        network_wins += wins
        print(
            f"  {result['input_snr_db']:>+5g} dB {result['detector']:16} "
            f"input+clean {result['reference_median']:.2f} "
            f"best {result['best_condition']}/{result['best_cleaning']} "
            f"{result['level_median']:.2f} diff {r3.fmt(result['median_diff'])} "
            f"p_holm {result['p_holm']:.2g}"
        )
    print(
        f"  Q3 decision: the network is significantly higher in {network_wins} of "
        f"{len(q3)} cells -> 'no network beats a standard filter' "
        f"{'stands' if network_wins == 0 else 'FAILS'}"
    )
    r3.write("r5_q3_filter_baseline.csv", q3)

    # Q4 — cleaning the network's own output.
    q4: list[Record] = []
    for level in levels:
        for name in conditions:
            for detector in DETECTORS:
                raw_values = output_f1(name, level, detector, "raw")
                if not raw_values:
                    continue
                result = r3.compare(
                    raw_values,
                    output_f1(name, level, detector, "ecg_clean"),
                    r3.F1_MARGIN,
                )
                result.update(
                    {"input_snr_db": level, "condition": name, "detector": detector}
                )
                q4.append(result)
    r3.write("r5_q4_output_cleaning.csv", q4)
    print("\n  Q4 — cleaning the network output, median F1 change (neurokit):")
    for result in q4:
        if result["detector"] == "neurokit":
            print(
                f"  {result['input_snr_db']:>+5g} dB {result['condition']:16} "
                f"raw {result['reference_median']:.2f} -> clean "
                f"{result['level_median']:.2f} ({r3.fmt(result['median_diff'])})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=["encoders", "detectors", "all"], default="all"
    )
    args = parser.parse_args()
    if args.part in ("encoders", "all"):
        analyse_encoders()
    if args.part in ("detectors", "all"):
        analyse_detectors()


if __name__ == "__main__":
    main()
