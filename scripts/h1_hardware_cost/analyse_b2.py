"""
H1 grid B2 analysis — the tests `b2_grid.py` states, and the front they move.

Every rule applied here is in `b2_grid.py`'s docstring, written before the grid
ran (P2b amended while it ran, before any row was analysed; the docstring says
why). Statistics are R3's: records are units, SNR averages a record's `ok` runs,
paired tests pair records, Wilcoxon with Holm where a family is named, BCa
intervals on medians.

Outputs, in `results/h1_hardware_cost/analysis/` (the figure in `figures/`):
  b2_cells.csv   one row per design point: ceiling, costs, ok fraction
  b2_tests.csv   P1, P2a, P2b, F1 and F2, one row per test
  b2_front.csv   the MIT-BIH fronts (synaptic operations, spikes) over pilot B's
                 points and B2's
  b2_post_hoc.csv  the labelled post hoc section below
  b2_front.pdf

Post hoc, descriptive, designed after the tests above were read: the share of
output noise the 1/r_0 law attributes to encoder counting at 40 Hz, with the
rate-independent remainder Lambda_rest; and selected B2 points against the best
ceiling measured before B2 (R4b `[512,512]`, K = 16, 20 Hz), paired by record
over R4b's 10 seeds and B2's 5, TOST +/-0.5 dB.

Usage:
    uv run python scripts/h1_hardware_cost/analyse_b2.py
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
from neuroflux_core.metrics.cascade import (  # noqa: E402
    affine_to_excess_db,
    db_to_linear,
    excess_to_affine_db,
    linear_to_db,
)
from neuroflux_datasets.analysis import bootstrap_median_ci  # noqa: E402
from scipy import stats  # type: ignore[import-untyped]  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_r3 as r3  # noqa: E402
import b2_grid  # noqa: E402
import pilot_b_cost as pilot  # noqa: E402
from figures_r3 import (  # noqa: E402
    BLUE,
    IEEE_RC,
    INK,
    REFERENCE,
    SINGLE_COLUMN_IN,
    VERMILLION,
)

RESULTS = Path("results/h1_hardware_cost")
R4B_FAN_IN = Path("results/r4b_mechanism/fan_in.csv")
LAW_MAE_DB = 0.3
FRAME_GAIN_DB = 0.5
NON_INFERIORITY_DB = -0.5

Row = dict[str, str]
Record = dict[str, Any]
Cell = tuple[int, int, float]  # (N_in, N_1, encoder target Hz)


def encoder_target(row: Row) -> float:
    for part in row["spec_key"].split("|"):
        if part.startswith("encoder_rate_hz="):
            return float(part.split("=", 1)[1])
    raise ValueError(f"No encoder rate in {row['spec_key']}")


def load_cells() -> dict[Cell, list[Row]]:
    path = pilot.source(b2_grid.OUTPUT)
    if path is None:
        raise SystemExit(f"{b2_grid.OUTPUT} not found; run b2_grid.py first.")
    cells: dict[Cell, list[Row]] = defaultdict(list)
    for row in r3.load(path):
        n_in, n_1 = (int(w) for w in row["layers"].split("-"))
        cells[(n_in, n_1, encoder_target(row))].append(row)
    return cells


def ceilings(cells: dict[Cell, list[Row]], cell: Cell) -> dict[str, float]:
    values: dict[str, float] = r3.unit_values(cells[cell], "snr_affine_db")
    return values


def nominal_synops(cell: Cell) -> float:
    _, n_1, encoder_hz = cell
    return encoder_hz * b2_grid.FAN_IN * n_1


def label(cell: Cell) -> str:
    n_in, n_1, encoder_hz = cell
    return f"[{n_in},{n_1}] K={b2_grid.FAN_IN} enc {encoder_hz:g} Hz"


# --------------------------------------------------------------------------
# Check and tests
# --------------------------------------------------------------------------


def reproduction_check(cells: dict[Cell, list[Row]]) -> Record:
    """B2's [128,128] 40 Hz cell against R4b's identical runs, seeds 0-4."""
    path = pilot.source(R4B_FAN_IN)
    reference: dict[tuple[str, str], float] = {}
    if path is not None:
        for row in r3.load(path):
            if (
                row["layers"] == "128-128"
                and row["fan_in"] == "8"
                and row["layer_rate_targets_hz"] == "20"
            ):
                reference[(row["unit"], row["seed"])] = float(row["snr_affine_db"])
    diffs = [
        abs(float(row["snr_affine_db"]) - reference[(row["unit"], row["seed"])])
        for row in cells[(128, 128, 40.0)]
        if (row["unit"], row["seed"]) in reference
    ]
    record: Record = {
        "test": "check: [128,128] 40 Hz reproduces R4b",
        "n_matched": len(diffs),
        "max_abs_diff_db": max(diffs) if diffs else math.nan,
        "verdict": "reproduced" if diffs and max(diffs) == 0.0 else "DIFFERS",
    }
    print(
        f"  Check: {len(diffs)} runs matched against R4b, max |diff| "
        f"{record['max_abs_diff_db']:.3g} dB -> {record['verdict']}"
    )
    return record


def test_p1(cells: dict[Cell, list[Row]]) -> list[Record]:
    family: list[Record] = []
    for n_1 in b2_grid.LAYER_WIDTHS:
        by_width = {
            n_in: ceilings(cells, (n_in, n_1, 40.0)) for n_in in b2_grid.INPUT_WIDTHS
        }
        units = sorted(set.intersection(*[set(v) for v in by_width.values()]))
        x = np.log2(np.array(b2_grid.INPUT_WIDTHS, dtype=float))
        slopes = [
            float(np.polyfit(x, [by_width[w][u] for w in b2_grid.INPUT_WIDTHS], 1)[0])
            for u in units
        ]
        low, high = bootstrap_median_ci(slopes, seed=0)
        medians = {w: r3.median(by_width[w].values()) for w in b2_grid.INPUT_WIDTHS}
        print(
            f"  N_1 = {n_1}, 40 Hz encoder, ceiling by N_in: "
            + ", ".join(f"{w}: {m:.2f}" for w, m in medians.items())
        )
        family.append(
            {
                "test": f"P1 slope on log2(N_in), N_1 = {n_1}",
                "n_units": len(slopes),
                "median_diff": r3.median(slopes),
                "ci_low": low,
                "ci_high": high,
                "p_value": float(stats.wilcoxon(slopes).pvalue),
            }
        )
    r3.holm(family)
    supported = all(r["median_diff"] > 0 and r["significant"] for r in family)
    for r in family:
        r["verdict"] = "supported" if supported else "not supported"
        print(
            f"  {r['test']}: {r3.fmt(r['median_diff'], 3)} dB/doubling "
            f"[{r3.fmt(r['ci_low'], 3)}, {r3.fmt(r['ci_high'], 3)}], "
            f"p_holm {r['p_holm']:.2g} -> {r['verdict']}"
        )
    return family


def test_f1(cells: dict[Cell, list[Row]]) -> Record:
    record: Record = r3.compare(
        ceilings(cells, (128, 128, 40.0)),
        ceilings(cells, (2048, 128, 40.0)),
        r3.SNR_MARGIN,
    )
    met = record["median_diff"] >= FRAME_GAIN_DB and record["p_value"] < r3.ALPHA
    record.update(
        {"test": "F1 [2048,128] minus [128,128], 40 Hz", "verdict": _met(met)}
    )
    print(
        f"  {record['test']}: {r3.fmt(record['median_diff'])} dB "
        f"[{r3.fmt(record['ci_low'])}, {r3.fmt(record['ci_high'])}], "
        f"p {record['p_value']:.2g} -> {record['verdict']}"
    )
    return record


def test_p2a(cells: dict[Cell, list[Row]]) -> list[Record]:
    family: list[Record] = []
    for n_1 in b2_grid.LAYER_WIDTHS:
        for n_in in b2_grid.INPUT_WIDTHS:
            record: Record = r3.compare(
                ceilings(cells, (n_in, n_1, 10.0)),
                ceilings(cells, (n_in, n_1, 40.0)),
                r3.SNR_MARGIN,
            )
            record.update(
                {
                    "test": f"P2a 40 Hz minus 10 Hz encoder, [{n_in},{n_1}]",
                    "db_per_halving": record["median_diff"] / 2.0,
                }
            )
            family.append(record)
    r3.holm(family)
    supported = all(r["median_diff"] > 0 and r["significant"] for r in family)
    for r in family:
        r["verdict"] = "supported" if supported else "not supported"
        print(
            f"  {r['test']}: {r3.fmt(r['median_diff'])} dB "
            f"[{r3.fmt(r['ci_low'])}, {r3.fmt(r['ci_high'])}] "
            f"({r3.fmt(r['db_per_halving'])} per halving), p_holm {r['p_holm']:.2g}"
        )
    print(f"  P2a -> {family[0]['verdict']}")
    return family


def inverse_excess(affine_db: float) -> float:
    """1 / (S/N) for an affine-corrected SNR; inf when it carries no signal."""
    excess = affine_to_excess_db(affine_db)
    return math.inf if excess == -math.inf else 1.0 / db_to_linear(excess)


def test_p2b(cells: dict[Cell, list[Row]]) -> Record:
    errors: list[float] = []
    signed: list[float] = []
    excluded = 0
    for n_1 in b2_grid.LAYER_WIDTHS:
        for n_in in b2_grid.INPUT_WIDTHS:
            at = {hz: ceilings(cells, (n_in, n_1, hz)) for hz in (10.0, 20.0, 40.0)}
            for unit in sorted(set(at[10.0]) & set(at[20.0]) & set(at[40.0])):
                y20, y40 = (
                    inverse_excess(at[20.0][unit]),
                    inverse_excess(at[40.0][unit]),
                )
                if not (math.isfinite(y20) and math.isfinite(y40)):
                    excluded += 1
                    continue
                c = (y20 - y40) / (1.0 / 20.0 - 1.0 / 40.0)
                y10 = c / 10.0 + (y40 - c / 40.0)
                if y10 <= 0.0:
                    errors.append(math.inf)
                    continue
                predicted = excess_to_affine_db(linear_to_db(1.0 / y10))
                signed.append(at[10.0][unit] - predicted)
                errors.append(abs(signed[-1]))
    # np.median, not r3.median: an unbounded error must count, not be dropped.
    mae = float(np.median(errors)) if errors else math.nan
    record: Record = {
        "test": "P2b law predicts 10 Hz from 20 and 40 Hz, all cells",
        "n_cells": len(errors),
        "n_excluded": excluded,
        "n_unbounded": sum(1 for e in errors if math.isinf(e)),
        "median_abs_error_db": mae,
        "median_signed_error_db": r3.median(signed),
        "p90_abs_error_db": float(np.percentile(errors, 90)) if errors else math.nan,
        "verdict": "supported" if mae <= LAW_MAE_DB else "not supported",
    }
    print(
        f"  {record['test']}: median |error| "
        f"{record['median_abs_error_db']:.3f} dB (signed "
        f"{record['median_signed_error_db']:+.3f}, 90th pct "
        f"{record['p90_abs_error_db']:.3f}) over {record['n_cells']} cells, "
        f"{record['n_unbounded']} unbounded, {excluded} excluded -> "
        f"{record['verdict']}"
    )
    return record


def test_f2(cells: dict[Cell, list[Row]]) -> list[Record]:
    reference_cell: Cell = (128, 128, 40.0)
    reference = ceilings(cells, reference_cell)
    cost = nominal_synops(reference_cell)
    records: list[Record] = []
    superiority: list[Record] = []
    for cell in sorted(cells):
        if cell == reference_cell:
            continue
        if nominal_synops(cell) <= 0.5 * cost:
            kind = "non-inferiority"
        elif nominal_synops(cell) == cost:
            kind = "superiority"
        else:
            continue
        record: Record = r3.compare(reference, ceilings(cells, cell), r3.SNR_MARGIN)
        record.update(
            {
                "test": f"F2 {kind}: {label(cell)} vs [128,128] enc 40 Hz",
                "kind": kind,
                "synops_ratio": nominal_synops(cell) / cost,
            }
        )
        if kind == "superiority":
            superiority.append(record)
        records.append(record)
    r3.holm(superiority)
    for r in records:
        if r["kind"] == "non-inferiority":
            r["verdict"] = _met(r["ci_low"] > NON_INFERIORITY_DB)
        else:
            r["verdict"] = _met(r["median_diff"] > 0 and r["significant"])
        print(
            f"  {r['test']}: {r3.fmt(r['median_diff'])} dB "
            f"[{r3.fmt(r['ci_low'])}, {r3.fmt(r['ci_high'])}] at "
            f"{r['synops_ratio']:.3g}x synops -> {r['verdict']}"
        )
    return records


def _met(condition: bool) -> str:
    return "met" if condition else "not met"


# --------------------------------------------------------------------------
# Costs, front, figure
# --------------------------------------------------------------------------


def post_hoc(cells: dict[Cell, list[Row]]) -> list[Record]:
    records: list[Record] = []
    print("\nPost hoc — encoder counting noise at 40 Hz (1/r_0 law)")
    for n_1 in b2_grid.LAYER_WIDTHS:
        for n_in in b2_grid.INPUT_WIDTHS:
            at = {hz: ceilings(cells, (n_in, n_1, hz)) for hz in (20.0, 40.0)}
            shares: list[float] = []
            rest_db: list[float] = []
            for unit in sorted(set(at[20.0]) & set(at[40.0])):
                y20, y40 = (
                    inverse_excess(at[20.0][unit]),
                    inverse_excess(at[40.0][unit]),
                )
                c = (y20 - y40) / (1.0 / 20.0 - 1.0 / 40.0)
                rest = y40 - c / 40.0
                shares.append((c / 40.0) / y40)
                rest_db.append(linear_to_db(1.0 / rest) if rest > 0 else math.inf)
            record: Record = {
                "analysis": "encoder noise share at 40 Hz",
                "design": f"[{n_in},{n_1}] K={b2_grid.FAN_IN}",
                "encoder_share_median": float(np.median(shares)),
                "lambda_rest_excess_db": float(np.median(rest_db)),
                "lambda_rest_affine_db": excess_to_affine_db(float(np.median(rest_db))),
            }
            records.append(record)
            print(
                f"  {record['design']:<16} encoder share "
                f"{record['encoder_share_median']:.2f}, Lambda_rest "
                f"{record['lambda_rest_affine_db']:.2f} dB affine"
            )
    path = pilot.source(R4B_FAN_IN)
    if path is None:
        return records
    best_rows = [
        r
        for r in r3.load(path)
        if r["layers"] == "512-512"
        and r["fan_in"] == "16"
        and r["layer_rate_targets_hz"] == "20"
    ]
    best = r3.unit_values(best_rows, "snr_affine_db")
    print("\nPost hoc — against the best earlier ceiling, [512,512] K=16 at 20 Hz")
    for cell in ((512, 128, 40.0), (2048, 128, 40.0), (2048, 128, 20.0)):
        record = r3.compare(best, ceilings(cells, cell), r3.SNR_MARGIN)
        record.update(
            {"analysis": "against [512,512] K=16 20 Hz", "design": label(cell)}
        )
        records.append(record)
        print(
            f"  {label(cell)} minus best: {r3.fmt(record['median_diff'])} dB "
            f"[{r3.fmt(record['ci_low'])}, {r3.fmt(record['ci_high'])}], "
            f"p {record['p_value']:.2g}, equivalent {record['equivalent']}"
        )
    return records


def cell_records(cells: dict[Cell, list[Row]]) -> list[Record]:
    records: list[Record] = []
    for cell, rows in sorted(cells.items()):
        n_in, n_1, encoder_hz = cell
        ok = [r for r in rows if r["verdict"] == "ok"]
        synops: dict[str, list[float]] = defaultdict(list)
        spikes: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            r_0 = float(r["encoder_rate_hz"])
            r_1 = float(r["layer_rates_hz"])
            synops[r3.unit_id(r)].append(r_0 * b2_grid.FAN_IN * n_1)
            spikes[r3.unit_id(r)].append(n_in * r_0 + n_1 * r_1)
        records.append(
            {
                "signal": "mit-bih",
                "design": label(cell),
                "lever": "B2",
                "n_in": n_in,
                "n_1": n_1,
                "encoder_hz": encoder_hz,
                "ceiling_db": r3.median(ceilings(cells, cell).values()),
                "synops_per_s": r3.median(float(np.mean(v)) for v in synops.values()),
                "synapses": float(b2_grid.FAN_IN * n_1),
                "spikes_per_s": r3.median(float(np.mean(v)) for v in spikes.values()),
                "ok_fraction": len(ok) / len(rows),
                "n_units": len(ceilings(cells, cell)),
            }
        )
    return records


def pilot_points() -> list[Record]:
    path = pilot.ANALYSIS / "pilot_b_points.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["signal"] == "mit-bih"]
    return [
        {
            "design": r["design"],
            "lever": r["lever"],
            "ceiling_db": float(r["ceiling_db"]),
            "synops_per_s": float(r["synops_per_s"]),
            "synapses": float(r["synapses"]),
            "spikes_per_s": float(r["spikes_per_s"]),
        }
        for r in rows
    ]


def figure(b2: list[Record], earlier: list[Record], front: list[Record]) -> None:
    mpl.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * SINGLE_COLUMN_IN, 2.3), layout="constrained"
    )
    ax = axes[0]
    colours = {32: VERMILLION, 128: BLUE}
    styles = {40.0: "-", 20.0: "--", 10.0: ":"}
    for n_1 in b2_grid.LAYER_WIDTHS:
        for hz in b2_grid.ENCODER_RATES_HZ:
            points = sorted(
                (r["n_in"], r["ceiling_db"])
                for r in b2
                if r["n_1"] == n_1 and r["encoder_hz"] == hz
            )
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=colours[n_1],
                ls=styles[hz],
                marker="o",
                ms=2.5,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Input pool $N_\\mathrm{in}$ (fan-in fixed at 8)")
    ax.set_ylabel("Ceiling (dB, noiseless input)")
    ax.set_title("(a) Input pool and encoder rate", loc="left")
    handles = [
        Line2D([], [], color=colours[n], label=f"$N_1$ = {n}")
        for n in b2_grid.LAYER_WIDTHS
    ] + [
        Line2D([], [], color=INK, ls=styles[hz], label=f"encoder {hz:g} Hz")
        for hz in sorted(b2_grid.ENCODER_RATES_HZ, reverse=True)
    ]
    ax.legend(handles=handles, loc="lower right", ncol=1, fontsize=6)

    ax = axes[1]
    ax.scatter(
        [r["synops_per_s"] for r in earlier],
        [r["ceiling_db"] for r in earlier],
        s=8,
        color=REFERENCE,
        edgecolors="white",
        linewidths=0.3,
        label="R3 and R4b design points",
        zorder=2,
    )
    for r in b2:
        ax.scatter(
            r["synops_per_s"],
            r["ceiling_db"],
            s=11,
            marker="D",
            color=colours[r["n_1"]],
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    ax.step(
        [r["synops_per_s"] for r in front],
        [r["ceiling_db"] for r in front],
        where="post",
        color=INK,
        lw=0.6,
        ls="--",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Synaptic operations per second")
    ax.set_title("(b) Synaptic-operation front", loc="left")
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color=REFERENCE, label="R3, R4b"),
            Line2D([], [], ls="", marker="D", color=BLUE, label="B2, $N_1$ = 128"),
            Line2D([], [], ls="", marker="D", color=VERMILLION, label="B2, $N_1$ = 32"),
            Line2D([], [], color=INK, lw=0.6, ls="--", label="Pareto front"),
        ],
        loc="lower right",
        fontsize=6,
    )
    stem = pilot.FIGURES / "b2_front"
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
    print(f"H1 grid B2: {runs} runs in {len(cells)} cells, {ok} ok")

    tests: list[Record] = [reproduction_check(cells)]
    print("\nP1 — input pool at fixed fan-in")
    tests += test_p1(cells)
    print("\nF1 — frame gain")
    f1 = test_f1(cells)
    tests.append(f1)
    print("\nP2a — encoder rate")
    tests += test_p2a(cells)
    print("\nP2b — the law prices the encoder rate")
    tests.append(test_p2b(cells))
    print("\nF2 — does any B2 point dominate [128,128] K=8, 40 Hz encoder?")
    f2 = test_f2(cells)
    tests += f2
    frame = f1["verdict"] == "met" or any(r["verdict"] == "met" for r in f2)
    print(
        f"\nFrame decision: F1 {f1['verdict']}, F2 "
        f"{'met' if any(r['verdict'] == 'met' for r in f2) else 'not met'} -> "
        + ("cost against fidelity IS the frame" if frame else "NOT the frame")
    )
    r3.write("b2_tests.csv", tests)

    b2 = cell_records(cells)
    r3.write("b2_cells.csv", b2)
    earlier = pilot_points()
    front = pilot.pareto(earlier + b2, "synops_per_s")
    print("\nMIT-BIH synaptic-operation front, R3 + R4b + B2:")
    for r in front:
        print(
            f"  {r['design']:<34} {r['ceiling_db']:5.2f} dB  "
            f"{r['synops_per_s']:>10.4g} synops/s"
        )
    spike_front = pilot.pareto(earlier + b2, "spikes_per_s")
    print("\nMIT-BIH spike front, R3 + R4b + B2:")
    for r in spike_front:
        print(
            f"  {r['design']:<34} {r['ceiling_db']:5.2f} dB  "
            f"{r['spikes_per_s']:>10.4g} spikes/s"
        )
    r3.write(
        "b2_front.csv",
        [{"cost": "synops_per_s", **r} for r in front]
        + [{"cost": "spikes_per_s", **r} for r in spike_front],
    )
    r3.write("b2_post_hoc.csv", post_hoc(cells))
    figure(b2, earlier, front)


if __name__ == "__main__":
    main()
