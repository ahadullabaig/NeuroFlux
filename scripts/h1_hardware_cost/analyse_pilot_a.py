"""
H1 pilot A analysis — the tests `pilot_a_grid.py` states.

Every rule is in `pilot_a_grid.py`'s docstring, committed with the per-record
predictions (`analysis/pilot_a_predictions.csv`) before the grid ran. PA3 reads
those committed predictions; nothing here recomputes or refits them. Statistics
are R3's: records are units, SNR averages a record's `ok` runs, paired tests
pair records, TOST at +/-0.5 dB, BCa intervals.

Outputs, in `results/h1_hardware_cost/analysis/`:
  pilot_a_cells.csv   one row per design point: ceiling, change from exact
                      weights (paired), predicted change, ok fraction
  pilot_a_tests.csv   the check, PA1-PA4
  pilot_a_post_hoc.csv  post hoc, descriptive: rank correlation and slope of
                      measured on predicted change over the 99 perturbed points,
                      overall and per pool
  ../figures/pilot_a.pdf

Usage:
    uv run python scripts/h1_hardware_cost/analyse_pilot_a.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_r3 as r3  # noqa: E402
import b2_grid  # noqa: E402
import pilot_a_grid as grid  # noqa: E402
import pilot_b_cost as pilot  # noqa: E402
from figures_r3 import (  # noqa: E402
    BLUE,
    GREEN,
    IEEE_RC,
    INK,
    REFERENCE,
    SINGLE_COLUMN_IN,
    VERMILLION,
)

LAW_MAE_DB = 0.3
BIG_LOSS_DB = -0.5
MIN_OK_FRACTION = 0.5

Row = dict[str, str]
Record = dict[str, Any]
Cell = tuple[int, float, str]  # (N_in, encoder Hz, level name)


def key_fields(row: Row) -> dict[str, str]:
    return dict(part.split("=", 1) for part in row["spec_key"].split("|"))


def level_name(fields: dict[str, str]) -> str:
    wanted = (
        int(fields["weight_bits"]) if "weight_bits" in fields else None,
        float(fields["weight_sigma"]) if "weight_sigma" in fields else None,
        float(fields["weight_stuck"]) if "weight_stuck" in fields else None,
    )
    return next(n for n, level in grid.levels().items() if level == wanted)


def load_cells() -> dict[Cell, list[Row]]:
    path = pilot.source(grid.OUTPUT)
    if path is None:
        raise SystemExit(f"{grid.OUTPUT} not found; run pilot_a_grid.py first.")
    cells: dict[Cell, list[Row]] = defaultdict(list)
    for row in r3.load(path):
        fields = key_fields(row)
        n_in = int(row["layers"].split("-")[0])
        cell = (n_in, float(fields["encoder_rate_hz"]), level_name(fields))
        cells[cell].append(row)
    return cells


def ceilings(cells: dict[Cell, list[Row]], cell: Cell) -> dict[str, float]:
    values: dict[str, float] = r3.unit_values(cells[cell], "snr_affine_db")
    return values


def ok_fraction(cells: dict[Cell, list[Row]], cell: Cell) -> float:
    rows = cells[cell]
    return sum(r["verdict"] == "ok" for r in rows) / len(rows)


def points() -> list[tuple[int, float]]:
    return [(n, hz) for n in grid.INPUT_WIDTHS for hz in grid.ENCODER_RATES_HZ]


def fmt_point(n_in: int, hz: float) -> str:
    return f"[{n_in},{grid.LAYER_WIDTH}] enc {hz:g} Hz"


# --------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------


def reproduction_check(cells: dict[Cell, list[Row]]) -> Record:
    path = pilot.source(b2_grid.OUTPUT)
    reference: dict[tuple[int, float, str, str], float] = {}
    if path is not None:
        for row in r3.load(path):
            n_in, n_1 = (int(w) for w in row["layers"].split("-"))
            if n_1 == grid.LAYER_WIDTH:
                hz = float(key_fields(row)["encoder_rate_hz"])
                reference[(n_in, hz, row["unit"], row["seed"])] = float(
                    row["snr_affine_db"]
                )
    diffs = [
        abs(float(r["snr_affine_db"]) - reference[(n, hz, r["unit"], r["seed"])])
        for (n, hz, name), rows in cells.items()
        if name == "exact"
        for r in rows
        if (n, hz, r["unit"], r["seed"]) in reference
    ]
    record: Record = {
        "test": "check: exact weights reproduce B2",
        "n_matched": len(diffs),
        "max_abs_diff_db": max(diffs) if diffs else math.nan,
        "verdict": "reproduced" if diffs and max(diffs) == 0.0 else "DIFFERS",
    }
    print(
        f"  Check: {len(diffs)} runs matched against B2, max |diff| "
        f"{record['max_abs_diff_db']:.3g} dB -> {record['verdict']}"
    )
    return record


# --------------------------------------------------------------------------
# PA1, PA2 — budgets
# --------------------------------------------------------------------------


def equivalent_to_exact(
    cells: dict[Cell, list[Row]], n_in: int, hz: float, name: str
) -> Record:
    record: Record = r3.compare(
        ceilings(cells, (n_in, hz, "exact")),
        ceilings(cells, (n_in, hz, name)),
        r3.SNR_MARGIN,
    )
    return record


def test_pa1(cells: dict[Cell, list[Row]]) -> list[Record]:
    records: list[Record] = []
    for n_in, hz in points():
        equivalent = {
            b: bool(equivalent_to_exact(cells, n_in, hz, f"{b} bit")["equivalent"])
            for b in grid.BITS
        }
        b_min: int | None = None
        for b in grid.BITS:
            if all(equivalent[x] for x in grid.BITS if x >= b):
                b_min = b
                break
        records.append(
            {
                "test": f"PA1 b_min, {fmt_point(n_in, hz)}",
                "b_min": b_min if b_min is not None else "> 4",
                "met": b_min is not None and b_min <= 2,
            }
        )
    supported = all(r["met"] for r in records)
    for r in records:
        r["verdict"] = "supported" if supported else "not supported"
    print(
        "  PA1 b_min: "
        + "; ".join(f"{r['test'][9:]} {r['b_min']}" for r in records)
        + f" -> {records[0]['verdict']}"
    )
    return records


def test_pa2(cells: dict[Cell, list[Row]]) -> list[Record]:
    records: list[Record] = []
    for n_in, hz in points():
        sigma_max = 0.0
        for sigma in grid.SIGMAS:
            if equivalent_to_exact(cells, n_in, hz, f"sigma {sigma:g}")["equivalent"]:
                sigma_max = sigma
            else:
                break
        records.append(
            {
                "test": f"PA2 sigma_max, {fmt_point(n_in, hz)}",
                "sigma_max": sigma_max,
                "met": sigma_max >= 0.5,
            }
        )
    supported = all(r["met"] for r in records)
    for r in records:
        r["verdict"] = "supported" if supported else "not supported"
    print(
        "  PA2 sigma_max: "
        + "; ".join(f"{r['test'][14:]} {r['sigma_max']:g}" for r in records)
        + f" -> {records[0]['verdict']}"
    )
    return records


# --------------------------------------------------------------------------
# PA3 — the committed predictions
# --------------------------------------------------------------------------


def committed_predictions() -> tuple[
    dict[tuple[int, float, str, str], float], dict[Cell, float]
]:
    per_record: dict[tuple[int, float, str, str], float] = {}
    with (pilot.ANALYSIS / "pilot_a_predictions.csv").open(encoding="utf-8") as h:
        for r in csv.DictReader(h):
            per_record[
                (int(r["n_in"]), float(r["encoder_hz"]), r["level"], r["unit"])
            ] = float(r["predicted_level_db"])
    median_change: dict[Cell, float] = {}
    with (pilot.ANALYSIS / "pilot_a_prediction_summary.csv").open(
        encoding="utf-8"
    ) as h:
        for r in csv.DictReader(h):
            median_change[(int(r["n_in"]), float(r["encoder_hz"]), r["level"])] = float(
                r["median_predicted_change_db"]
            )
    return per_record, median_change


def test_pa3(cells: dict[Cell, list[Row]]) -> Record:
    predicted, median_change = committed_predictions()
    model_errors: list[float] = []
    big_model: list[float] = []
    big_null: list[float] = []
    excluded_points = 0
    missing_units = 0
    big_points = 0
    for n_in, hz in points():
        exact = ceilings(cells, (n_in, hz, "exact"))
        for name in grid.levels():
            if name == "exact":
                continue
            cell = (n_in, hz, name)
            if ok_fraction(cells, cell) < MIN_OK_FRACTION:
                excluded_points += 1
                continue
            observed = ceilings(cells, cell)
            big = median_change[cell] <= BIG_LOSS_DB
            big_points += big
            for unit, value in observed.items():
                key = (n_in, hz, name, unit)
                if key not in predicted or unit not in exact:
                    missing_units += 1
                    continue
                error = abs(value - predicted[key])
                model_errors.append(error)
                if big:
                    big_model.append(error)
                    big_null.append(abs(value - exact[unit]))
    mae = float(np.median(model_errors))
    big_mae = float(np.median(big_model)) if big_model else math.nan
    null_mae = float(np.median(big_null)) if big_null else math.nan
    part_i = mae <= LAW_MAE_DB
    part_ii = big_mae < null_mae
    record: Record = {
        "test": "PA3 budget predicts the device losses",
        "n_values": len(model_errors),
        "median_abs_error_db": mae,
        "big_loss_points": big_points,
        "big_model_median_abs_error_db": big_mae,
        "big_null_median_abs_error_db": null_mae,
        "excluded_points": excluded_points,
        "missing_units": missing_units,
        "part_i": part_i,
        "part_ii": part_ii,
        "verdict": "supported" if part_i and part_ii else "not supported",
    }
    print(
        f"  PA3 (i) median |error| {mae:.3f} dB over {len(model_errors)} values "
        f"-> {'met' if part_i else 'not met'}"
    )
    print(
        f"  PA3 (ii) on {big_points} points predicted to lose >= 0.5 dB: model "
        f"{big_mae:.3f} dB, no-change null {null_mae:.3f} dB -> "
        f"{'met' if part_ii else 'not met'}"
    )
    print(
        f"  PA3 -> {record['verdict']} ({excluded_points} points excluded, "
        f"{missing_units} record values missing)"
    )
    return record


# --------------------------------------------------------------------------
# PA4 — matched effective fan-in
# --------------------------------------------------------------------------


def test_pa4(cells: dict[Cell, list[Row]]) -> list[Record]:
    records: list[Record] = []
    for narrow, wide in grid.MATCHED_PAIRS:
        for n_in, hz in points():
            record: Record = r3.compare(
                ceilings(cells, (n_in, hz, narrow)),
                ceilings(cells, (n_in, hz, wide)),
                r3.SNR_MARGIN,
            )
            record["test"] = f"PA4 {wide} minus {narrow}, {fmt_point(n_in, hz)}"
            records.append(record)
    supported = all(r["equivalent"] for r in records)
    for r in records:
        r["verdict"] = "supported" if supported else "not supported"
        print(
            f"  {r['test']}: {r3.fmt(r['median_diff'])} dB "
            f"[{r3.fmt(r['ci_low'])}, {r3.fmt(r['ci_high'])}], "
            f"equivalent {r['equivalent']}"
        )
    print(f"  PA4 -> {records[0]['verdict']}")
    return records


# --------------------------------------------------------------------------
# Cells and figure
# --------------------------------------------------------------------------


def cell_records(cells: dict[Cell, list[Row]]) -> list[Record]:
    _, median_change = committed_predictions()
    records: list[Record] = []
    for n_in, hz in points():
        for name in grid.levels():
            cell = (n_in, hz, name)
            change = equivalent_to_exact(cells, n_in, hz, name)
            records.append(
                {
                    "n_in": n_in,
                    "n_1": grid.LAYER_WIDTH,
                    "encoder_hz": hz,
                    "level": name,
                    "ceiling_db": r3.median(ceilings(cells, cell).values()),
                    "change_db": change["median_diff"],
                    "change_ci_low": change["ci_low"],
                    "change_ci_high": change["ci_high"],
                    "equivalent_to_exact": change["equivalent"],
                    "predicted_change_db": median_change[cell],
                    "ok_fraction": ok_fraction(cells, cell),
                }
            )
    return records


KIND_STYLE: dict[str, tuple[str, str, str]] = {
    "bit": (BLUE, "o", "quantized"),
    "sigma": (VERMILLION, "s", "lognormal variation"),
    "stuck": (GREEN, "^", "stuck at zero"),
}


def figure(records: list[Record]) -> None:
    mpl.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * SINGLE_COLUMN_IN, 2.4), layout="constrained"
    )
    ax = axes[0]
    low = min(min(r["predicted_change_db"], r["change_db"]) for r in records) - 0.1
    ax.plot([low, 0.1], [low, 0.1], color=REFERENCE, lw=0.6, ls="--", zorder=1)
    for r in records:
        if r["level"] == "exact":
            continue
        kind = next(k for k in KIND_STYLE if k in r["level"])
        colour, marker, _ = KIND_STYLE[kind]
        ax.scatter(
            r["predicted_change_db"],
            r["change_db"],
            s=10,
            marker=marker,
            color=colour,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    ax.set_xlabel("Predicted change (dB, committed)")
    ax.set_ylabel("Measured change from exact weights (dB)")
    ax.set_title("(a) Budget model against measurement", loc="left")
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker=m, color=c, label=label)
            for c, m, label in KIND_STYLE.values()
        ]
        + [Line2D([], [], color=REFERENCE, ls="--", lw=0.6, label="identity")],
        loc="lower right",
        fontsize=6,
    )

    ax = axes[1]
    styles = {10.0: ":", 40.0: "-"}
    colours = {128: VERMILLION, 512: BLUE, 2048: INK}
    for n_in in grid.INPUT_WIDTHS:
        for hz, ls in styles.items():
            xs = [0.0, *[p for p in grid.STUCK]]
            ys = [
                next(
                    r["change_db"]
                    for r in records
                    if r["n_in"] == n_in
                    and r["encoder_hz"] == hz
                    and r["level"] == ("exact" if p == 0.0 else f"stuck {p:g}")
                )
                for p in xs
            ]
            ax.plot(
                [100 * x for x in xs],
                ys,
                color=colours[n_in],
                ls=ls,
                marker="o",
                ms=2.5,
                label=f"$N_\\mathrm{{in}}$ = {n_in}, {hz:g} Hz",
            )
    ax.set_xlabel("Devices stuck at zero (%)")
    ax.set_ylabel("Measured change (dB)")
    ax.set_title("(b) Open-device faults", loc="left")
    ax.legend(loc="lower left", fontsize=5.5, ncol=2)
    stem = pilot.FIGURES / "pilot_a"
    pilot.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.pdf (+ .png preview)")


def main() -> None:
    r3.OUT = pilot.ANALYSIS
    cells = load_cells()
    runs = sum(len(v) for v in cells.values())
    ok = sum(r["verdict"] == "ok" for v in cells.values() for r in v)
    print(f"H1 pilot A: {runs} runs in {len(cells)} design points, {ok} ok")

    tests: list[Record] = [reproduction_check(cells)]
    print("\nPA1 — precision budget")
    tests += test_pa1(cells)
    print("\nPA2 — variation budget")
    tests += test_pa2(cells)
    print("\nPA3 — does the budget price the faults?")
    tests.append(test_pa3(cells))
    print("\nPA4 — is effective fan-in the whole story?")
    tests += test_pa4(cells)
    r3.write("pilot_a_tests.csv", tests)

    records = cell_records(cells)
    r3.write("pilot_a_cells.csv", records)
    print("\nMeasured change from exact weights, median over records (predicted):")
    for n_in, hz in points():
        line = ", ".join(
            f"{r['level']} {r['change_db']:+.2f} ({r['predicted_change_db']:+.2f})"
            for r in records
            if r["n_in"] == n_in and r["encoder_hz"] == hz and r["level"] != "exact"
        )
        print(f"  {fmt_point(n_in, hz)}: {line}")
    post_hoc(records)
    figure(records)


def post_hoc(records: list[Record]) -> None:
    """Designed after PA3 was read: how the committed predictions miss."""
    from scipy import stats  # type: ignore[import-untyped]

    rows: list[Record] = []
    perturbed = [r for r in records if r["level"] != "exact"]
    groups: list[tuple[str, list[Record]]] = [("all", perturbed)] + [
        (f"N_in = {n}", [r for r in perturbed if r["n_in"] == n])
        for n in grid.INPUT_WIDTHS
    ]
    for name, group in groups:
        x = np.array([r["predicted_change_db"] for r in group])
        y = np.array([r["change_db"] for r in group])
        rows.append(
            {
                "analysis": "measured on predicted change (post hoc)",
                "group": name,
                "n_points": len(group),
                "spearman": float(stats.spearmanr(x, y).statistic),
                "slope_through_origin": float(x @ y / (x @ x)),
            }
        )
        print(
            f"  post hoc {name}: Spearman {rows[-1]['spearman']:.2f}, "
            f"slope {rows[-1]['slope_through_origin']:.2f}"
        )
    r3.write("pilot_a_post_hoc.csv", rows)


if __name__ == "__main__":
    main()
