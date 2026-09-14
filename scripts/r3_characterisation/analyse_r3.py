"""
R3 analysis — the pre-registered statistics for C1, C2 and C3.

Every rule applied here is stated in the docstring of the script that produced
the data (`c1_flatness.py`, `c2_invariances.py`, `c3_operating_region.py`) and
was written before that grid ran. This file implements those rules and nothing
else; a number that needs a new rule belongs in a new, labelled section.

Shared conventions:

  * The unit of analysis is a record on MIT-BIH and a seed on synthetic signals.
  * SNR metrics average a unit's `ok` runs only — a silent or saturated output
    has no reconstruction to score. QRS F1 averages every run, because a silent
    output detecting nothing is a result.
  * Paired tests pair units. Wilcoxon signed-rank; Holm-Bonferroni within the
    family named in the output; BCa bootstrap interval on the median
    difference; TOST at ±0.5 dB for SNR and ±0.05 for F1.

Usage:
    uv run python scripts/r3_characterisation/analyse_r3.py            # all parts
    uv run python scripts/r3_characterisation/analyse_r3.py --part c1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from neuroflux_core.metrics.cascade import (
    affine_to_excess_db,
    cascade_snr_db,
    excess_to_affine_db,
    intrinsic_snr_db,
)
from neuroflux_datasets.analysis import (
    bootstrap_median_ci,
    holm_bonferroni,
    tost_equivalence,
)
from scipy import stats  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import split_floats  # noqa: E402

RESULTS = Path("results/r3_characterisation")
OUT = RESULTS / "analysis"
SNR_MARGIN = 0.5
F1_MARGIN = 0.05
ALPHA = 0.05

Row = dict[str, str]
Record = dict[str, Any]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def fnum(value: str) -> float:
    """A CSV field as a float; empty fields are NaN."""
    return float(value) if value else math.nan


def raw(name: str) -> Path:
    """
    A raw results file: the plain CSV a grid writes, or the gzipped copy that is
    committed in its place when the plain one is absent (a fresh clone).
    """
    plain = RESULTS / name
    packed = plain.with_name(plain.name + ".gz")
    return packed if not plain.exists() and packed.exists() else plain


def load(path: Path) -> list[Row]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unit_id(row: Row) -> str:
    """Records are units on MIT-BIH; seeds are units on synthetic signals."""
    return row["unit"] if row["signal"] == "mit-bih" else f"seed{row['seed']}"


def unit_values(rows: Iterable[Row], metric: str) -> dict[str, float]:
    """
    One value per unit: the mean of `metric` over the unit's runs.

    SNR-type metrics count `ok` runs only; `qrs_*` metrics count every run.
    Units with no finite value are dropped.
    """
    every_run = metric.startswith("qrs_")
    buckets: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not every_run and row["verdict"] != "ok":
            continue
        value = fnum(row[metric])
        if math.isfinite(value):
            buckets[unit_id(row)].append(value)
    return {unit: float(np.mean(values)) for unit, values in buckets.items()}


def median(values: Iterable[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


def compare(
    reference: dict[str, float],
    other: dict[str, float],
    margin: float,
) -> Record:
    """Paired comparison `other - reference` over the units both contain."""
    units = sorted(reference.keys() & other.keys())
    diffs = [other[u] - reference[u] for u in units]
    p_value = math.nan
    if len(diffs) >= 2 and any(d != 0.0 for d in diffs):
        p_value = float(stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
    low, high = bootstrap_median_ci(diffs, seed=0)
    tost = tost_equivalence(diffs, margin=margin, alpha=ALPHA)
    return {
        "n_units": len(diffs),
        "reference_median": median(reference[u] for u in units),
        "level_median": median(other[u] for u in units),
        "median_diff": median(diffs),
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "tost_p": tost["tost_p"],
        "equivalent": bool(tost["equivalent"]),
        "margin": margin,
    }


def holm(family: list[Record]) -> None:
    """Holm-adjust a family in place and mark significance."""
    adjusted = holm_bonferroni([r["p_value"] for r in family])
    for record, p_holm in zip(family, adjusted, strict=True):
        record["p_holm"] = p_holm
        record["family_size"] = len(family)
        record["significant"] = math.isfinite(p_holm) and p_holm < ALPHA


def verdict_of(family: list[Record]) -> str:
    """The pre-registered reading of one knob's family."""
    if family and all(r["equivalent"] for r in family):
        return "invariant"
    if any(r["significant"] and abs(r["median_diff"]) >= r["margin"] for r in family):
        return "matters"
    return "inconclusive"


def write(name: str, records: list[Record]) -> None:
    if not records:
        print(f"  (no rows for {name})")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        fieldnames.extend(k for k in record if k not in fieldnames)
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  wrote {len(records)} rows to {path}")


def law_prediction(input_affine_db: float, ceiling_affine_db: float) -> float:
    """
    Output affine SNR the cascade law predicts, applied in `S/N` terms.

    Affine-corrected SNR is `1 + S/N` (`metrics/cascade.py`), so the law is
    applied after `affine_to_excess_db` and converted back. No parameters.
    """
    r_in = affine_to_excess_db(input_affine_db)
    lam = affine_to_excess_db(ceiling_affine_db)
    if r_in == -math.inf or lam == -math.inf:
        return 0.0
    if r_in == math.inf:
        return ceiling_affine_db
    return excess_to_affine_db(cascade_snr_db(r_in, [lam]))


def intrinsic_excess_db(input_affine_db: float, output_affine_db: float) -> float:
    """A stage's intrinsic `S/N`, from the affine SNRs entering and leaving it."""
    r_in = affine_to_excess_db(input_affine_db)
    r_out = affine_to_excess_db(output_affine_db)
    if not (math.isfinite(r_in) and math.isfinite(r_out)):
        return math.nan
    return intrinsic_snr_db(r_in, r_out)


def select(rows: Iterable[Row], **criteria: str) -> list[Row]:
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def fmt(value: float, digits: int = 2) -> str:
    return "  nan" if not math.isfinite(value) else f"{value:+.{digits}f}"


# --------------------------------------------------------------------------
# C1 — flatness
# --------------------------------------------------------------------------

C1_ARMS: tuple[tuple[str, str], ...] = (
    ("controlled", "0"),
    ("shipped", "0"),
    ("shipped", "10"),
)
SLOPE_RANGES: dict[str, tuple[float, float]] = {
    "full": (-20.0, 40.0),
    "high": (10.0, 40.0),
}


def arm_label(operating_point: str, epochs: str) -> str:
    return f"{operating_point}-{'trained' if epochs != '0' else 'passive'}"


def per_level(rows: list[Row], metric: str) -> dict[float, dict[str, float]]:
    """Unit values at every nominal input SNR."""
    levels: defaultdict[float, list[Row]] = defaultdict(list)
    for row in rows:
        levels[fnum(row["input_snr_nominal"])].append(row)
    return {level: unit_values(group, metric) for level, group in levels.items()}


def unit_slopes(
    curve: dict[float, dict[str, float]], low: float, high: float
) -> dict[str, float]:
    """Least-squares slope per unit over [low, high], needing three points."""
    # Sorted, not a set: the bootstrap interval depends on sample order, and set
    # order changes between processes with string hash randomisation.
    units = sorted({u for values in curve.values() for u in values})
    slopes: dict[str, float] = {}
    for unit in units:
        points = [
            (level, values[unit])
            for level, values in curve.items()
            if low <= level <= high and unit in values and math.isfinite(level)
        ]
        if len(points) >= 3:
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            slopes[unit] = float(np.polyfit(x, y, 1)[0])
    return slopes


def analyse_c1() -> None:
    path = raw("c1_flatness.csv")
    if not path.exists():
        print(f"C1: {path} not found")
        return
    rows = load(path)
    print(f"\nC1 — flatness ({len(rows)} rows)")

    curve_records: list[Record] = []
    slope_records: list[Record] = []
    law_records: list[Record] = []
    interference_records: list[Record] = []

    for signal in ("synthetic-ecg", "mit-bih", "eeg", "sine"):
        for operating_point, epochs in C1_ARMS:
            arm = arm_label(operating_point, epochs)
            group = select(
                rows,
                signal=signal,
                operating_point=operating_point,
                epochs=epochs,
                interference="gaussian",
            )
            if not group:
                continue
            output = per_level(group, "snr_affine_db")
            raw_output = per_level(group, "snr_raw_db")
            reference = per_level(group, "input_affine_db")
            ceiling = output.get(math.inf, {})

            by_level: defaultdict[float, list[Row]] = defaultdict(list)
            for row in group:
                by_level[fnum(row["input_snr_nominal"])].append(row)

            for level in sorted(output):
                ok = [r["verdict"] == "ok" for r in by_level[level]]
                # Pre-registered: the law on affine values. Post hoc: the law
                # in S/N terms, which the metric's definition requires.
                residuals: dict[str, list[float]] = {"affine": [], "excess": []}
                predictions: dict[str, list[float]] = {"affine": [], "excess": []}
                if math.isfinite(level):
                    for unit, measured in output[level].items():
                        if unit in ceiling and unit in reference[level]:
                            for form, predicted in (
                                (
                                    "affine",
                                    cascade_snr_db(
                                        reference[level][unit], [ceiling[unit]]
                                    ),
                                ),
                                (
                                    "excess",
                                    law_prediction(
                                        reference[level][unit], ceiling[unit]
                                    ),
                                ),
                            ):
                                predictions[form].append(predicted)
                                residuals[form].append(measured - predicted)
                curve_records.append(
                    {
                        "signal": signal,
                        "arm": arm,
                        "input_snr_db": level,
                        "n_units": len(output[level]),
                        "ok_fraction": float(np.mean(ok)),
                        "input_affine_median": median(reference[level].values()),
                        "output_affine_median": median(output[level].values()),
                        "output_affine_q25": float(
                            np.percentile(list(output[level].values()), 25)
                        )
                        if output[level]
                        else math.nan,
                        "output_affine_q75": float(
                            np.percentile(list(output[level].values()), 75)
                        )
                        if output[level]
                        else math.nan,
                        "output_raw_median": median(raw_output.get(level, {}).values()),
                        "predicted_affine_law_median": median(predictions["affine"]),
                        "residual_affine_law_median": median(residuals["affine"]),
                        "predicted_median": median(predictions["excess"]),
                        "residual_median": median(residuals["excess"]),
                        "out_rate_median": median(
                            unit_values(by_level[level], "out_rate_hz").values()
                        ),
                    }
                )
                for form, values in residuals.items():
                    for value in values:
                        law_records.append(
                            {
                                "signal": signal,
                                "arm": arm,
                                "form": form,
                                "input_snr_db": level,
                                "residual": value,
                            }
                        )

            for name, (low, high) in SLOPE_RANGES.items():
                slopes = unit_slopes(output, low, high)
                reference_slopes = unit_slopes(reference, low, high)
                values = list(slopes.values())
                ci_low, ci_high = bootstrap_median_ci(values, seed=0)
                slope_records.append(
                    {
                        "signal": signal,
                        "arm": arm,
                        "range": name,
                        "n_units": len(values),
                        "slope_median": median(values),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "excludes_one": math.isfinite(ci_high) and ci_high < 1.0,
                        "input_slope_median": median(reference_slopes.values()),
                    }
                )

    print("  Slope of affine output SNR on input SNR (median, 95% BCa CI):")
    for r in slope_records:
        print(
            f"    {r['signal']:>13} {r['arm']:>18} {r['range']:>4}: "
            f"{fmt(r['slope_median'], 3)} "
            f"[{fmt(r['ci_low'], 3)}, {fmt(r['ci_high'], 3)}] "
            f"n={r['n_units']:<3} input {fmt(r['input_slope_median'], 3)} "
            f"{'excludes 1' if r['excludes_one'] else 'DOES NOT exclude 1'}"
        )

    law_summary: list[Record] = []
    for signal in ("synthetic-ecg", "mit-bih", "eeg", "sine"):
        for operating_point, epochs in C1_ARMS:
            arm = arm_label(operating_point, epochs)
            for form, regime in itertools.product(
                ("affine", "excess"), (">=10 dB", "<10 dB")
            ):
                resid = np.array(
                    [
                        r["residual"]
                        for r in law_records
                        if r["signal"] == signal
                        and r["arm"] == arm
                        and r["form"] == form
                        and (r["input_snr_db"] >= 10.0) == (regime == ">=10 dB")
                    ]
                )
                if resid.size == 0:
                    continue
                law_summary.append(
                    {
                        "signal": signal,
                        "arm": arm,
                        "form": (
                            "affine values (pre-registered)"
                            if form == "affine"
                            else "S/N (post hoc, parameter-free)"
                        ),
                        "regime": regime,
                        "n_points": int(resid.size),
                        "mean": float(resid.mean()),
                        "median": float(np.median(resid)),
                        "rms": float(np.sqrt(np.mean(resid**2))),
                        "max_abs": float(np.max(np.abs(resid))),
                    }
                )
    print("  Cascade-law residual, measured minus predicted (dB):")
    for r in law_summary:
        print(
            f"    {r['signal']:>13} {r['arm']:>18} {r['form'][:6]:>6} "
            f"{r['regime']:>7}: "
            f"mean {fmt(r['mean'])} "
            f"RMS {r['rms']:.2f} max {r['max_abs']:.2f} (n={r['n_points']})"
        )

    for signal in ("synthetic-ecg", "mit-bih"):
        for operating_point, epochs in (("controlled", "0"), ("shipped", "10")):
            arm = arm_label(operating_point, epochs)
            base = select(
                rows, signal=signal, operating_point=operating_point, epochs=epochs
            )
            gaussian = per_level(select(base, interference="gaussian"), "snr_affine_db")
            for kind in ("flicker", "mixed"):
                other = per_level(select(base, interference=kind), "snr_affine_db")
                family: list[Record] = []
                for level in sorted(other):
                    if level not in gaussian:
                        continue
                    record = compare(gaussian[level], other[level], SNR_MARGIN)
                    record.update(
                        {
                            "signal": signal,
                            "arm": arm,
                            "comparison": f"{kind} minus gaussian",
                            "input_snr_db": level,
                        }
                    )
                    family.append(record)
                holm(family)
                interference_records.extend(family)
    print("  Interference type at matched input SNR (median difference, dB):")
    for r in interference_records:
        tag = "equiv" if r["equivalent"] else ("SIG" if r["significant"] else "incon")
        print(
            f"    {r['signal']:>13} {r['arm']:>18} {r['comparison']:>22} "
            f"{r['input_snr_db']:>5.0f} dB: {fmt(r['median_diff'])} "
            f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] p_holm={r['p_holm']:.3g} {tag}"
        )

    write("c1_curve.csv", curve_records)
    write("c1_slopes.csv", slope_records)
    write("c1_law_residuals.csv", law_summary)
    write("c1_interference.csv", interference_records)


def analyse_c1_operating_point() -> None:
    """
    Post hoc: the S/N-form law with intrinsic SNRs measured at each input's own
    operating point (`c1_operating_point.py`), against the ordinary S/N form.
    """
    path = raw("c1_operating_point.csv")
    c1_path = raw("c1_flatness.csv")
    if not path.exists() or not c1_path.exists():
        print(f"C1 operating point: {path} not found")
        return
    matched_rows = load(path)
    c1_rows = load(c1_path)
    print(f"\nC1 (post hoc) — operating-point-matched law ({len(matched_rows)} rows)")

    records: list[Record] = []
    summary: list[Record] = []
    for signal, operating_point, interference in itertools.product(
        ("synthetic-ecg", "mit-bih", "eeg", "sine"),
        ("controlled", "shipped"),
        ("gaussian", "flicker"),
    ):
        arm = arm_label(operating_point, "0")
        matched_block = select(
            matched_rows,
            signal=signal,
            operating_point=operating_point,
            interference=interference,
        )
        if not matched_block:
            continue
        noisy = select(
            c1_rows,
            signal=signal,
            operating_point=operating_point,
            epochs="0",
            interference=interference,
        )
        output = per_level(noisy, "snr_affine_db")
        reference = per_level(noisy, "input_affine_db")
        # The ordinary ceiling is always the Gaussian arm's noiseless run:
        # C1 ran no noiseless input under a flicker label.
        ceiling = per_level(
            select(
                c1_rows,
                signal=signal,
                operating_point=operating_point,
                epochs="0",
                interference="gaussian",
            ),
            "snr_affine_db",
        ).get(math.inf, {})
        by_level: defaultdict[float, list[Row]] = defaultdict(list)
        for row in matched_block:
            by_level[fnum(row["calibrate_at_snr_db"])].append(row)
        residuals: dict[str, list[tuple[float, float]]] = {
            "ordinary": [],
            "matched": [],
        }
        for level, group in sorted(by_level.items()):
            matched_ceiling = unit_values(group, "snr_affine_db")
            shift: list[float] = []
            for unit, measured in output.get(level, {}).items():
                if unit not in reference[level]:
                    continue
                inp = reference[level][unit]
                if unit in ceiling:
                    residuals["ordinary"].append(
                        (level, measured - law_prediction(inp, ceiling[unit]))
                    )
                if unit in matched_ceiling:
                    residuals["matched"].append(
                        (
                            level,
                            measured - law_prediction(inp, matched_ceiling[unit]),
                        )
                    )
                    if unit in ceiling:
                        shift.append(matched_ceiling[unit] - ceiling[unit])
            records.append(
                {
                    "signal": signal,
                    "arm": arm,
                    "interference": interference,
                    "input_snr_db": level,
                    "ceiling_shift_median": median(shift),
                    "residual_ordinary_median": median(
                        r for lv, r in residuals["ordinary"] if lv == level
                    ),
                    "residual_matched_median": median(
                        r for lv, r in residuals["matched"] if lv == level
                    ),
                }
            )
        for form, values in residuals.items():
            for regime in (">=10 dB", "<10 dB"):
                arr = np.array(
                    [r for lv, r in values if (lv >= 10.0) == (regime == ">=10 dB")]
                )
                if arr.size == 0:
                    continue
                summary.append(
                    {
                        "signal": signal,
                        "arm": arm,
                        "interference": interference,
                        "intrinsics": form,
                        "regime": regime,
                        "n_points": int(arr.size),
                        "mean": float(arr.mean()),
                        "rms": float(np.sqrt(np.mean(arr**2))),
                        "max_abs": float(np.max(np.abs(arr))),
                    }
                )
    for r in summary:
        print(
            f"    {r['signal']:>13} {r['arm']:>18} {r['interference']:>8} "
            f"{r['intrinsics']:>8} "
            f"{r['regime']:>7}: mean {fmt(r['mean'])} RMS {r['rms']:.2f} "
            f"max {r['max_abs']:.2f} (n={r['n_points']})"
        )
    write("c1_operating_point_levels.csv", records)
    write("c1_operating_point_summary.csv", summary)


# --------------------------------------------------------------------------
# C2 — invariances
# --------------------------------------------------------------------------


def c2_knobs() -> dict[
    str, tuple[Callable[[Row], float | None], float, tuple[str, ...]]
]:
    """Knob name -> (level selector, reference level, operating points)."""

    def base(row: Row, *, epochs: str = "0", tau: str = "20.0") -> bool:
        return (
            row["epochs"] == epochs
            and row["stdp_tau"] == "20.0"
            and row["decoder_tau"] == tau
        )

    def output_width(row: Row) -> float | None:
        widths = row["layers"].split("-")
        if len(widths) == 3 and widths[:2] == ["32", "16"] and base(row):
            return float(widths[2])
        return None

    def decoder_tau(row: Row) -> float | None:
        if (
            row["layers"] == "32-16-8"
            and row["epochs"] == "0"
            and row["stdp_tau"] == "20.0"
        ):
            return fnum(row["decoder_tau"])
        return None

    def epochs(row: Row) -> float | None:
        if (
            row["layers"] == "32-16-8"
            and row["stdp_tau"] == "20.0"
            and row["decoder_tau"] == "20.0"
        ):
            return float(row["epochs"])
        return None

    def depth(row: Row) -> float | None:
        if row["layers"] in ("32-8", "32-16-8", "32-16-16-8") and base(row):
            return float(len(row["layers"].split("-")))
        return None

    def stdp_tau(row: Row) -> float | None:
        if (
            row["layers"] == "32-16-8"
            and row["epochs"] == "10"
            and row["decoder_tau"] == "20.0"
        ):
            return fnum(row["stdp_tau"])
        return None

    return {
        "output_width": (output_width, 8.0, ("controlled",)),
        "decoder_tau": (decoder_tau, 20.0, ("controlled",)),
        "epochs": (epochs, 0.0, ("controlled", "shipped")),
        "depth": (depth, 3.0, ("controlled",)),
        "stdp_tau": (stdp_tau, 20.0, ("controlled",)),
    }


def analyse_c2() -> None:
    path = raw("c2_invariances.csv")
    if not path.exists():
        print(f"C2: {path} not found")
        return
    rows = load(path)
    print(f"\nC2 — invariances ({len(rows)} rows)")

    comparisons: list[Record] = []
    verdicts: list[Record] = []
    levels_summary: list[Record] = []

    for knob, (selector, reference_level, operating_points) in c2_knobs().items():
        for signal in ("synthetic-ecg", "mit-bih"):
            for input_label, input_value in (("clean", "inf"), ("10 dB", "10.0")):
                for operating_point in operating_points:
                    group = select(
                        rows,
                        signal=signal,
                        input_snr_nominal=input_value,
                        operating_point=operating_point,
                    )
                    by_level: defaultdict[float, list[Row]] = defaultdict(list)
                    for row in group:
                        level = selector(row)
                        if level is not None:
                            by_level[level].append(row)
                    if reference_level not in by_level:
                        continue
                    metrics = [("snr_affine_db", SNR_MARGIN)]
                    if signal == "mit-bih":
                        metrics.append(("qrs_f1", F1_MARGIN))
                    for metric, margin in metrics:
                        reference = unit_values(by_level[reference_level], metric)
                        family: list[Record] = []
                        for level in sorted(by_level):
                            values = unit_values(by_level[level], metric)
                            levels_summary.append(
                                {
                                    "knob": knob,
                                    "signal": signal,
                                    "input": input_label,
                                    "operating_point": operating_point,
                                    "metric": metric,
                                    "level": level,
                                    "n_units": len(values),
                                    "median": median(values.values()),
                                    "ok_fraction": float(
                                        np.mean(
                                            [
                                                r["verdict"] == "ok"
                                                for r in by_level[level]
                                            ]
                                        )
                                    ),
                                    "out_rate_median": median(
                                        unit_values(
                                            by_level[level], "out_rate_hz"
                                        ).values()
                                    ),
                                }
                            )
                            if level == reference_level:
                                continue
                            record = compare(reference, values, margin)
                            record.update(
                                {
                                    "knob": knob,
                                    "signal": signal,
                                    "input": input_label,
                                    "operating_point": operating_point,
                                    "metric": metric,
                                    "reference_level": reference_level,
                                    "level": level,
                                }
                            )
                            family.append(record)
                        holm(family)
                        comparisons.extend(family)
                        verdicts.append(
                            {
                                "knob": knob,
                                "signal": signal,
                                "input": input_label,
                                "operating_point": operating_point,
                                "metric": metric,
                                "verdict": verdict_of(family),
                                "levels": len(family) + 1,
                                "largest_abs_median_diff": max(
                                    (abs(r["median_diff"]) for r in family),
                                    default=math.nan,
                                ),
                            }
                        )

    print("  Knob verdicts (pre-registered reading):")
    for v in verdicts:
        print(
            f"    {v['knob']:>13} {v['signal']:>13} {v['input']:>6} "
            f"{v['operating_point']:>10} "
            f"{v['metric']:>14}: {v['verdict']:<12} largest |median diff| "
            f"{v['largest_abs_median_diff']:.3f}"
        )

    width_records = c2_width_prediction(rows)
    write("c2_comparisons.csv", comparisons)
    write("c2_verdicts.csv", verdicts)
    write("c2_levels.csv", levels_summary)
    write("c2_width_prediction.csv", width_records)


def c2_width_prediction(rows: list[Row]) -> list[Record]:
    """
    Output width against the independent-neuron capacity prediction.

    Per unit, from its N = 8 reference run: the SNR entering the output layer
    (`S_in`, the hidden layer's stage SNR) and the output layer's intrinsic SNR
    `Lambda_8 = intrinsic(S_in, C_8)`. Independent neurons would give
    `Lambda_N = Lambda_8 * N / 8`, so the predicted ceiling is
    `cascade(S_in, [Lambda_N])`. Nothing is fitted. Every step is in `S/N`
    terms, converted from and back to affine SNR (`metrics/cascade.py`).
    """
    records: list[Record] = []
    for signal in ("synthetic-ecg", "mit-bih"):
        group = [
            r
            for r in select(
                rows,
                signal=signal,
                input_snr_nominal="inf",
                operating_point="controlled",
                epochs="0",
                stdp_tau="20.0",
                decoder_tau="20.0",
            )
            if r["layers"].startswith("32-16-")
            and len(r["layers"].split("-")) == 3
            and r["verdict"] == "ok"
        ]
        by_unit: defaultdict[str, defaultdict[int, list[tuple[float, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for row in group:
            stages = split_floats(row["stage_affine_db"])
            width = int(row["layers"].split("-")[2])
            by_unit[unit_id(row)][width].append((stages[-2], stages[-1]))

        fitted_c: list[float] = []
        for unit, widths in by_unit.items():
            if 8 not in widths:
                continue
            s_in = float(np.mean([s for s, _ in widths[8]]))
            c_ref = float(np.mean([c for _, c in widths[8]]))
            if not (math.isfinite(s_in) and math.isfinite(c_ref)):
                continue
            lambda_ref = intrinsic_excess_db(s_in, c_ref)
            inverse_lambdas: list[tuple[int, float]] = []
            for width in sorted(widths):
                s_w = float(np.mean([s for s, _ in widths[width]]))
                c_w = float(np.mean([c for _, c in widths[width]]))
                predicted = (
                    excess_to_affine_db(
                        cascade_snr_db(
                            affine_to_excess_db(s_in),
                            [lambda_ref + 10.0 * math.log10(width / 8)],
                        )
                    )
                    if math.isfinite(lambda_ref)
                    else math.nan
                )
                lam_w = intrinsic_excess_db(s_w, c_w)
                if math.isfinite(lam_w):
                    inverse_lambdas.append((width, 10 ** (-lam_w / 10.0)))
                records.append(
                    {
                        "signal": signal,
                        "unit": unit,
                        "width": width,
                        "measured_ceiling": c_w,
                        "capacity_predicted_ceiling": predicted,
                        "hidden_stage_snr": s_w,
                        "output_lambda": lam_w,
                    }
                )
            if len(inverse_lambdas) >= 3:
                n = np.array([w for w, _ in inverse_lambdas], dtype=float)
                y = np.array([v for _, v in inverse_lambdas])
                design = np.vstack([1.0 / n, 1.0 - 1.0 / n]).T
                (k, kc), *_ = np.linalg.lstsq(design, y, rcond=None)
                if k > 0:
                    fitted_c.append(float(kc / k))

        for width in sorted({r["width"] for r in records if r["signal"] == signal}):
            block = [
                r for r in records if r["signal"] == signal and r["width"] == width
            ]
            measured = median(r["measured_ceiling"] for r in block)
            predicted = median(r["capacity_predicted_ceiling"] for r in block)
            print(
                f"  width {signal:>13} N={width:<3} measured {measured:.2f} dB"
                f"  capacity prediction {predicted:.2f} dB"
            )
        if fitted_c:
            print(
                f"  width model fit, {signal}: median c = {np.median(fitted_c):.3f} "
                f"IQR [{np.percentile(fitted_c, 25):.3f}, "
                f"{np.percentile(fitted_c, 75):.3f}] "
                f"over {len(fitted_c)} units (descriptive)"
            )
            records.append(
                {
                    "signal": signal,
                    "unit": "summary",
                    "fitted_c_median": float(np.median(fitted_c)),
                }
            )
    return records


# --------------------------------------------------------------------------
# C3 — operating region
# --------------------------------------------------------------------------


def crossing(
    gains: list[float], fractions: list[float], threshold: float, falling: bool
) -> float:
    """First gain at which a fraction crosses `threshold`, interpolated in log gain."""
    for i, (g, f) in enumerate(zip(gains, fractions, strict=True)):
        hit = f <= threshold if falling else f >= threshold
        if not hit:
            continue
        if i == 0:
            return g
        g0, f0 = gains[i - 1], fractions[i - 1]
        if f == f0:
            return g
        t = (threshold - f0) / (f - f0)
        return float(math.exp(math.log(g0) + t * (math.log(g) - math.log(g0))))
    return math.nan


def analyse_c3_plane() -> None:
    path = raw("c3_plane.csv")
    if not path.exists():
        print(f"C3 plane: {path} not found")
        return
    rows = load(path)
    print(f"\nC3 (a) — gain plane ({len(rows)} rows)")
    cells: defaultdict[tuple[int, str, float], list[Row]] = defaultdict(list)
    for row in rows:
        cells[
            (int(row["layers"].split("-")[0]), row["epochs"], fnum(row["g_syn"]))
        ].append(row)

    cell_records: list[Record] = []
    for (width, epochs, gain), group in sorted(cells.items()):
        verdicts = [r["verdict"] for r in group]
        cell_records.append(
            {
                "n_in": width,
                "epochs": int(epochs),
                "g_syn": gain,
                "n_runs": len(group),
                "silent_fraction": verdicts.count("silent") / len(group),
                "saturated_fraction": verdicts.count("saturated") / len(group),
                "ok_fraction": verdicts.count("ok") / len(group),
                "snr_affine_median": median(
                    unit_values(group, "snr_affine_db").values()
                ),
                "qrs_f1_median": median(unit_values(group, "qrs_f1").values()),
                "out_rate_median": median(
                    float(
                        np.median(
                            [fnum(r["out_rate_hz"]) for r in group if unit_id(r) == u]
                        )
                    )
                    for u in {unit_id(r) for r in group}
                ),
                "layer1_rate_median": float(
                    np.median([split_floats(r["layer_rates_hz"])[0] for r in group])
                ),
            }
        )

    boundaries: list[Record] = []
    for epochs in sorted({r["epochs"] for r in cell_records}):
        for width in sorted({r["n_in"] for r in cell_records}):
            block = sorted(
                (
                    r
                    for r in cell_records
                    if r["n_in"] == width and r["epochs"] == epochs
                ),
                key=lambda r: r["g_syn"],
            )
            gains = [r["g_syn"] for r in block]
            silent = crossing(
                gains, [r["silent_fraction"] for r in block], 0.5, falling=True
            )
            saturated = crossing(
                gains, [r["saturated_fraction"] for r in block], 0.5, falling=False
            )
            usable = [r for r in block if r["ok_fraction"] >= 0.5]
            best_snr = max(usable, key=lambda r: r["snr_affine_median"], default=None)
            best_f1 = max(block, key=lambda r: r["qrs_f1_median"], default=None)
            band = (
                math.log2(saturated / silent)
                if math.isfinite(silent) and math.isfinite(saturated)
                else math.nan
            )
            shipped = min(block, key=lambda r: abs(math.log(r["g_syn"] / 30.0)))
            boundaries.append(
                {
                    "epochs": epochs,
                    "n_in": width,
                    "silent_boundary_g": silent,
                    "saturated_boundary_g": saturated,
                    "usable_band_octaves": band,
                    "no_usable_gain": math.isfinite(band) and band <= 0.0,
                    "best_snr_g": best_snr["g_syn"] if best_snr else math.nan,
                    "best_snr_db": best_snr["snr_affine_median"]
                    if best_snr
                    else math.nan,
                    "best_f1_g": best_f1["g_syn"] if best_f1 else math.nan,
                    "best_f1": best_f1["qrs_f1_median"] if best_f1 else math.nan,
                    "shipped_g_cell": shipped["g_syn"],
                    "shipped_g_ok_fraction": shipped["ok_fraction"],
                    "shipped_g_snr_db": shipped["snr_affine_median"],
                    "shipped_g_f1": shipped["qrs_f1_median"],
                }
            )
    print("  Boundaries (gain), usable band, and the shipped g = 30 cell:")
    for b in boundaries:
        print(
            f"    epochs={b['epochs']:>2} N_in={b['n_in']:>3}: "
            f"silent<= {b['silent_boundary_g']:7.2f}  "
            f"saturated>= {b['saturated_boundary_g']:7.2f}  "
            f"band {b['usable_band_octaves']:5.2f} oct  "
            f"best SNR {b['best_snr_db']:.2f} dB @ g={b['best_snr_g']:.1f}  "
            f"best F1 {b['best_f1']:.2f} "
            f"@ g={b['best_f1_g']:.1f}  | g~30: ok {b['shipped_g_ok_fraction']:.2f} "
            f"SNR {b['shipped_g_snr_db']:.2f} F1 {b['shipped_g_f1']:.2f}"
        )
    write("c3_plane_cells.csv", cell_records)
    write("c3_plane_boundaries.csv", boundaries)


def rate_arm(row: Row) -> list[tuple[str, float]]:
    """(scope label, output rate) memberships of one c3-rate row."""
    targets = split_floats(row["layer_rate_targets_hz"])
    network = row["layers"]
    memberships: list[tuple[str, float]] = []
    if all(t == 40.0 for t in targets[:-1]):
        memberships.append((f"output {network}", targets[-1]))
    if network == "32-16-8" and all(t == targets[0] for t in targets):
        memberships.append((f"all {network}", targets[0]))
    return memberships


def analyse_c3_rate(name: str = "c3_rate") -> None:
    path = raw(f"{name}.csv")
    if not path.exists():
        print(f"C3 rate: {path} not found")
        return
    rows = load(path)
    print(f"\nC3 rate axis, {name} ({len(rows)} rows)")

    arms: defaultdict[tuple[str, str, str, float], list[Row]] = defaultdict(list)
    for row in rows:
        for scope, rate in rate_arm(row):
            arms[(row["signal"], scope, row["input_snr_nominal"], rate)].append(row)

    summary: list[Record] = []
    comparisons: list[Record] = []
    law: list[Record] = []
    signals = sorted({k[0] for k in arms})
    scopes = sorted({k[1] for k in arms})
    inputs = sorted({k[2] for k in arms}, key=fnum)
    for signal in signals:
        for scope in scopes:
            rates = sorted({k[3] for k in arms if k[0] == signal and k[1] == scope})
            for input_value in inputs:
                for rate in rates:
                    group = arms[(signal, scope, input_value, rate)]
                    summary.append(
                        {
                            "signal": signal,
                            "scope": scope,
                            "input_snr_db": fnum(input_value),
                            "target_rate_hz": rate,
                            "n_units": len({unit_id(r) for r in group}),
                            "ok_fraction": float(
                                np.mean([r["verdict"] == "ok" for r in group])
                            ),
                            "out_rate_median": median(
                                unit_values(group, "out_rate_hz").values()
                            ),
                            "snr_affine_median": median(
                                unit_values(group, "snr_affine_db").values()
                            ),
                            "qrs_f1_median": median(
                                unit_values(group, "qrs_f1").values()
                            )
                            if signal == "mit-bih"
                            else math.nan,
                            "qrs_f1_input_median": median(
                                unit_values(group, "qrs_f1_input").values()
                            )
                            if signal == "mit-bih"
                            else math.nan,
                        }
                    )
                metrics = [("snr_affine_db", SNR_MARGIN)]
                if signal == "mit-bih":
                    metrics.append(("qrs_f1", F1_MARGIN))
                for metric, margin in metrics:
                    reference_rows = arms.get((signal, scope, input_value, 40.0))
                    if not reference_rows:
                        continue
                    reference = unit_values(reference_rows, metric)
                    family: list[Record] = []
                    for rate in rates:
                        if rate == 40.0:
                            continue
                        record = compare(
                            reference,
                            unit_values(
                                arms[(signal, scope, input_value, rate)], metric
                            ),
                            margin,
                        )
                        record.update(
                            {
                                "signal": signal,
                                "scope": scope,
                                "input_snr_db": fnum(input_value),
                                "metric": metric,
                                "rate_hz": rate,
                                "reference_rate_hz": 40.0,
                            }
                        )
                        family.append(record)
                    holm(family)
                    comparisons.extend(family)

            for rate in rates:
                ceiling = unit_values(
                    arms.get((signal, scope, "inf", rate), []), "snr_affine_db"
                )
                for input_value in inputs:
                    if input_value == "inf":
                        continue
                    group = arms[(signal, scope, input_value, rate)]
                    measured = unit_values(group, "snr_affine_db")
                    input_affine = unit_values(group, "input_affine_db")
                    residuals = [
                        measured[u] - law_prediction(input_affine[u], ceiling[u])
                        for u in measured
                        if u in ceiling and u in input_affine
                    ]
                    law.append(
                        {
                            "signal": signal,
                            "scope": scope,
                            "rate_hz": rate,
                            "input_snr_db": fnum(input_value),
                            "n_units": len(residuals),
                            "residual_median": median(residuals),
                            "residual_rms": float(
                                np.sqrt(np.mean(np.square(residuals)))
                            )
                            if residuals
                            else math.nan,
                        }
                    )

    print("  Median over units by target output rate (ceiling = noiseless input):")
    for signal in signals:
        for scope in scopes:
            print(f"    {signal} / {scope}")
            for input_value in inputs:
                block = [
                    s
                    for s in summary
                    if s["signal"] == signal
                    and s["scope"] == scope
                    and s["input_snr_db"] == fnum(input_value)
                ]
                line = "  ".join(
                    f"{s['target_rate_hz']:g}Hz:{s['snr_affine_median']:.2f}"
                    + (
                        f"/F1 {s['qrs_f1_median']:.2f}"
                        if math.isfinite(s["qrs_f1_median"])
                        else ""
                    )
                    for s in block
                )
                print(f"      in={input_value:>4}  {line}")
    write(f"{name}_summary.csv", summary)
    write(f"{name}_comparisons.csv", comparisons)
    write(f"{name}_law.csv", law)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=[
            "c1",
            "c1-operating-point",
            "c2",
            "c3-plane",
            "c3-rate",
            "c3-rate-signals",
            "all",
        ],
        default="all",
    )
    args = parser.parse_args()
    if args.part in ("c1", "all"):
        analyse_c1()
    if args.part in ("c1-operating-point", "all"):
        analyse_c1_operating_point()
    if args.part in ("c2", "all"):
        analyse_c2()
    if args.part in ("c3-plane", "all"):
        analyse_c3_plane()
    if args.part in ("c3-rate", "all"):
        analyse_c3_rate()
    if args.part in ("c3-rate-signals", "all"):
        analyse_c3_rate("c3_rate_signals")


if __name__ == "__main__":
    main()
