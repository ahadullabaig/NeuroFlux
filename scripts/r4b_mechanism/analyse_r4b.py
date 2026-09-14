"""
R4b analysis — the pre-registered tests for next actions 13, 14 and 17.

Each rule is stated in the docstring of the grid script that produced the data
(`training_fan_in.py`, `encoder_range.py`, `rate_optimum.py`), written before
that grid ran. The statistics helpers are R3's (`analyse_r3.py`): records are
units on MIT-BIH and seeds elsewhere, SNR averages `ok` runs, paired tests pair
units, Wilcoxon + Holm + BCa + TOST.

Usage:
    uv run python scripts/r4b_mechanism/analyse_r4b.py
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from neuroflux_datasets.analysis import bootstrap_median_ci
from scipy import stats  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
import analyse_r3 as r3  # noqa: E402
from r3_common import split_floats  # noqa: E402

RESULTS = Path("results/r4b_mechanism")
r3.OUT = RESULTS / "analysis"

Record = dict[str, Any]


def source(name: str) -> Path | None:
    """The plain CSV a grid wrote, or its committed gzipped copy."""
    plain = RESULTS / name
    packed = plain.with_name(plain.name + ".gz")
    if plain.exists():
        return plain
    return packed if packed.exists() else None


# --------------------------------------------------------------------------
# Next action 17 — effective fan-in under training
# --------------------------------------------------------------------------


def analyse_training() -> None:
    path = source("training_fan_in.csv")
    if path is None:
        print("training_fan_in.csv not found")
        return
    rows = r3.load(path)
    print(f"\nNext action 17 — training and effective fan-in ({len(rows)} rows)")

    levels: list[Record] = []
    correlations: list[Record] = []
    for signal in ("mit-bih", "synthetic-ecg"):
        per_unit: defaultdict[str, defaultdict[int, list[dict[str, str]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for row in rows:
            if row["signal"] == signal and row["verdict"] == "ok":
                per_unit[r3.unit_id(row)][int(row["epochs"])].append(row)

        def unit_stats(group: list[dict[str, str]]) -> dict[str, float]:
            stages = np.mean(
                [split_floats(g["stage_affine_db"]) for g in group], axis=0
            )
            return {
                "ceiling": float(np.mean([float(g["ceiling_db"]) for g in group])),
                "pr1": float(
                    np.mean([float(g["participation_layer1"]) for g in group])
                ),
                "pr2": float(
                    np.mean([float(g["participation_layer2"]) for g in group])
                ),
                "cos1": float(np.mean([float(g["cosine_layer1"]) for g in group])),
                "cos2": float(np.mean([float(g["cosine_layer2"]) for g in group])),
                "lambda1": r3.intrinsic_excess_db(float(stages[0]), float(stages[1])),
                "lambda2": r3.intrinsic_excess_db(float(stages[1]), float(stages[2])),
            }

        table = {
            unit: {epochs: unit_stats(group) for epochs, group in by_epoch.items()}
            for unit, by_epoch in per_unit.items()
        }
        print(f"  {signal}: medians over {len(table)} units")
        print(
            f"    {'epochs':>6} {'ceiling':>8} {'PR L1':>7} {'PR L2':>7} "
            f"{'cos L1':>7} {'cos L2':>7} {'S/N L1':>7} {'S/N L2':>7}"
        )
        for epochs in (0, 10, 50, 200):
            med = {
                key: r3.median(t[epochs][key] for t in table.values() if epochs in t)
                for key in (
                    "ceiling",
                    "pr1",
                    "pr2",
                    "cos1",
                    "cos2",
                    "lambda1",
                    "lambda2",
                )
            }
            levels.append({"signal": signal, "epochs": epochs, **med})
            print(
                f"    {epochs:>6} {med['ceiling']:>8.2f} {med['pr1']:>7.2f} "
                f"{med['pr2']:>7.2f} {med['cos1']:>7.3f} {med['cos2']:>7.3f} "
                f"{med['lambda1']:>+7.2f} {med['lambda2']:>+7.2f}"
            )

        # P1: paired test of layer-1 PR, 0 against 200 epochs.
        pairs = [
            (t[0]["pr1"], t[200]["pr1"]) for t in table.values() if 0 in t and 200 in t
        ]
        p1 = (
            float(stats.wilcoxon([b - a for a, b in pairs]).pvalue)
            if len(pairs) > 1
            else math.nan
        )
        print(
            f"    P1 layer-1 PR, 0 vs 200 epochs: Wilcoxon p = {p1:.2g} "
            f"(n = {len(pairs)})"
        )

        for layer in ("1", "2"):
            d_pr, d_cos, d_lambda = [], [], []
            for t in table.values():
                for epochs in (10, 50, 200):
                    if 0 not in t or epochs not in t:
                        continue
                    values = (
                        t[epochs][f"pr{layer}"] - t[0][f"pr{layer}"],
                        t[epochs][f"cos{layer}"] - t[0][f"cos{layer}"],
                        t[epochs][f"lambda{layer}"] - t[0][f"lambda{layer}"],
                    )
                    if all(math.isfinite(v) for v in values):
                        d_pr.append(values[0])
                        d_cos.append(values[1])
                        d_lambda.append(values[2])
            rho_pr, p_pr = stats.spearmanr(d_pr, d_lambda)
            rho_cos, p_cos = stats.spearmanr(d_cos, d_lambda)
            correlations.append(
                {
                    "signal": signal,
                    "layer": int(layer),
                    "n_pairs": len(d_lambda),
                    "rho_participation": float(rho_pr),
                    "p_participation": float(p_pr),
                    "rho_cosine": float(rho_cos),
                    "p_cosine": float(p_cos),
                }
            )
            print(
                f"    layer {layer}: Spearman(change in intrinsic S/N, change in PR) "
                f"= {rho_pr:+.3f} (p = {p_pr:.2g}); with change in cosine "
                f"{rho_cos:+.3f} (p = {p_cos:.2g}); n = {len(d_lambda)}"
            )
    r3.write("r4b_training_levels.csv", levels)
    r3.write("r4b_training_correlations.csv", correlations)


# --------------------------------------------------------------------------
# Next action 14 — encoder range
# --------------------------------------------------------------------------


def analyse_encoder_range() -> None:
    path = source("encoder_range.csv")
    if path is None:
        print("encoder_range.csv not found")
        return
    rows = r3.load(path)
    print(f"\nNext action 14 — encoder range ({len(rows)} rows)")

    comparisons: list[Record] = []
    for signal in ("mit-bih", "synthetic-ecg", "sine"):
        block = r3.select(rows, signal=signal)
        by_method: defaultdict[str, defaultdict[float, list[dict[str, str]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for row in block:
            by_method[row["encoder_range"]][r3.fnum(row["input_snr_nominal"])].append(
                row
            )
        inputs = sorted(by_method["minmax"])
        reference = {
            level: r3.unit_values(by_method["minmax"][level], "snr_affine_db")
            for level in inputs
        }
        oracle = {
            level: r3.unit_values(by_method["oracle"][level], "snr_affine_db")
            for level in inputs
        }
        for method in ("percentile", "lowpass", "oracle"):
            family: list[Record] = []
            for level in inputs:
                values = r3.unit_values(by_method[method][level], "snr_affine_db")
                record = r3.compare(reference[level], values, r3.SNR_MARGIN)
                oracle_gain = r3.median(
                    oracle[level][u] - reference[level][u]
                    for u in reference[level]
                    if u in oracle[level]
                )
                record.update(
                    {
                        "signal": signal,
                        "method": method,
                        "input_snr_db": level,
                        "oracle_gain_median": oracle_gain,
                        "share_of_oracle_gain": record["median_diff"] / oracle_gain
                        if method != "oracle"
                        and oracle_gain
                        and math.isfinite(oracle_gain)
                        else math.nan,
                    }
                )
                family.append(record)
            r3.holm(family)
            comparisons.extend(family)

    for r in comparisons:
        tag = "equiv" if r["equivalent"] else ("SIG" if r["significant"] else "incon")
        print(
            f"  {r['signal']:>13} {r['method']:>10} in={r['input_snr_db']:>5}: "
            f"{r['reference_median']:.2f} -> {r['level_median']:.2f} "
            f"d {r3.fmt(r['median_diff'])} "
            f"[{r3.fmt(r['ci_low'])}, {r3.fmt(r['ci_high'])}] "
            f"p_holm {r['p_holm']:.2g} {tag}  "
            f"share of oracle {r['share_of_oracle_gain']:.2f}"
        )

    print("  Primary test (MIT-BIH, 0 dB input; ceiling at noiseless input):")
    for method in ("percentile", "lowpass"):
        at_zero = next(
            r
            for r in comparisons
            if r["signal"] == "mit-bih"
            and r["method"] == method
            and r["input_snr_db"] == 0.0
        )
        ceiling = next(
            r
            for r in comparisons
            if r["signal"] == "mit-bih"
            and r["method"] == method
            and math.isinf(r["input_snr_db"])
        )
        raises = at_zero["significant"] and at_zero["median_diff"] >= 0.5
        keeps = ceiling["equivalent"] or (
            ceiling["significant"] and ceiling["median_diff"] > 0
        )
        print(
            f"    {method:>10}: 0 dB gain {at_zero['median_diff']:+.2f} "
            f"(significant {at_zero['significant']}, "
            f">= +0.5 {at_zero['median_diff'] >= 0.5}); "
            f"ceiling change {ceiling['median_diff']:+.2f} (kept {keeps}) -> "
            f"{'LEVER ESTABLISHED' if raises and keeps else 'not established'}"
        )
    r3.write("r4b_encoder_range.csv", comparisons)


# --------------------------------------------------------------------------
# Next action 13 — width and fan-in at the rate optimum
# --------------------------------------------------------------------------


def analyse_width() -> None:
    path = source("width.csv")
    if path is None:
        print("width.csv not found")
        return
    rows = r3.load(path)
    print(
        f"\nNext action 13 (width) — input width at a sparse output ({len(rows)} rows)"
    )
    slopes: dict[float, dict[str, float]] = {}
    summary: list[Record] = []
    for rate in (5.0, 40.0):
        per_unit: defaultdict[str, dict[int, float]] = defaultdict(dict)
        for width in sorted({int(r["layers"].split("-")[0]) for r in rows}):
            group = [
                r
                for r in rows
                if int(r["layers"].split("-")[0]) == width
                and split_floats(r["layer_rate_targets_hz"])[-1] == rate
            ]
            values = r3.unit_values(group, "snr_affine_db")
            for unit, value in values.items():
                per_unit[unit][width] = value
            summary.append(
                {
                    "output_rate_hz": rate,
                    "n_in": width,
                    "ceiling_median": r3.median(values.values()),
                    "ok_fraction": float(
                        np.mean([r["verdict"] == "ok" for r in group])
                    ),
                }
            )
        slopes[rate] = {
            unit: float(
                np.polyfit(np.log2(list(by_width)), list(by_width.values()), 1)[0]
            )
            for unit, by_width in per_unit.items()
            if len(by_width) >= 3
        }
        low, high = bootstrap_median_ci(list(slopes[rate].values()), seed=0)
        print(
            f"  output {rate:g} Hz: median ceiling by N_in "
            + " ".join(
                f"{s['n_in']}:{s['ceiling_median']:.2f}"
                for s in summary
                if s["output_rate_hz"] == rate
            )
            + f"  | slope {r3.median(slopes[rate].values()):+.3f} dB/doubling "
            f"[{low:+.3f}, {high:+.3f}]"
        )
    units = sorted(slopes[5.0].keys() & slopes[40.0].keys())
    diffs = [slopes[5.0][u] - slopes[40.0][u] for u in units]
    p = float(stats.wilcoxon(diffs).pvalue)
    low, high = bootstrap_median_ci(diffs, seed=0)
    print(
        f"  slope at 5 Hz minus 40 Hz: median {r3.median(diffs):+.3f} "
        f"[{low:+.3f}, {high:+.3f}], "
        f"Wilcoxon p = {p:.2g}, n = {len(diffs)} -> "
        f"{'SUPPORTED' if p < 0.05 and r3.median(diffs) > 0 else 'not supported'}"
    )
    r3.write("r4b_width.csv", summary)


def analyse_fan_in() -> None:
    path = source("fan_in.csv")
    if path is None:
        print("fan_in.csv not found")
        return
    rows = r3.load(path)
    print(
        f"\nNext action 13 (fan-in) — M4's intervention against output rate "
        f"({len(rows)} rows)"
    )
    summary: list[Record] = []
    family: list[Record] = []
    for width in sorted({int(r["layers"].split("-")[0]) for r in rows}):
        cells: dict[tuple[int, float], dict[str, float]] = {}
        for fan_in in sorted(
            {int(r["fan_in"]) for r in rows if r["layers"].startswith(f"{width}-")}
        ):
            for rate in sorted(
                {split_floats(r["layer_rate_targets_hz"])[0] for r in rows}
            ):
                group = [
                    r
                    for r in rows
                    if r["layers"] == f"{width}-{width}"
                    and int(r["fan_in"]) == fan_in
                    and split_floats(r["layer_rate_targets_hz"])[0] == rate
                ]
                values = r3.unit_values(group, "snr_affine_db")
                cells[(fan_in, rate)] = values
                summary.append(
                    {
                        "width": width,
                        "fan_in": fan_in or "dense",
                        "output_rate_hz": rate,
                        "ceiling_median": r3.median(values.values()),
                        "out_rate_median": r3.median(
                            r3.unit_values(group, "out_rate_hz").values()
                        ),
                        "ok_fraction": float(
                            np.mean([r["verdict"] == "ok" for r in group])
                        ),
                    }
                )
        sparse_k = max(k for k, _ in cells)
        rates = sorted({rate for _, rate in cells})
        best_dense = max(rates, key=lambda rt: r3.median(cells[(0, rt)].values()))
        best_sparse = max(
            rates, key=lambda rt: r3.median(cells[(sparse_k, rt)].values())
        )
        line = " ".join(
            f"{rt:g}Hz:{r3.median(cells[(0, rt)].values()):.2f}/"
            f"{r3.median(cells[(sparse_k, rt)].values()):.2f}"
            for rt in rates
        )
        print(f"  [{width},{width}] dense/K={sparse_k} ceiling by layer rate: {line}")
        for rt in rates:
            gain = r3.compare(cells[(0, rt)], cells[(sparse_k, rt)], r3.SNR_MARGIN)
            print(
                f"    {rt:>4g} Hz fan-in gain {r3.fmt(gain['median_diff'])} "
                f"[{r3.fmt(gain['ci_low'])}, {r3.fmt(gain['ci_high'])}] "
                f"p {gain['p_value']:.2g}"
            )
        record = r3.compare(
            cells[(0, best_dense)], cells[(sparse_k, best_sparse)], r3.SNR_MARGIN
        )
        record.update(
            {
                "width": width,
                "fan_in": sparse_k,
                "best_dense_rate_hz": best_dense,
                "best_sparse_rate_hz": best_sparse,
            }
        )
        family.append(record)
    r3.holm(family)
    for r in family:
        verdict = (
            "SUPPORTED"
            if r["significant"] and r["median_diff"] >= 0.5
            else "not supported"
        )
        print(
            f"  Prediction, [{r['width']},{r['width']}]: K={r['fan_in']} "
            f"at its best rate "
            f"({r['best_sparse_rate_hz']:g} Hz) minus dense at its best "
            f"({r['best_dense_rate_hz']:g} Hz): {r3.fmt(r['median_diff'])} "
            f"[{r3.fmt(r['ci_low'])}, {r3.fmt(r['ci_high'])}], "
            f"p_holm {r['p_holm']:.2g} "
            f"-> {verdict}"
        )
    r3.write("r4b_fan_in_cells.csv", summary)
    r3.write("r4b_fan_in_test.csv", family)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=["training", "encoder-range", "width", "fan-in", "all"],
        default="all",
    )
    args = parser.parse_args()
    if args.part in ("training", "all"):
        analyse_training()
    if args.part in ("encoder-range", "all"):
        analyse_encoder_range()
    if args.part in ("width", "all"):
        analyse_width()
    if args.part in ("fan-in", "all"):
        analyse_fan_in()


if __name__ == "__main__":
    main()
