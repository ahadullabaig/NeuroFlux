"""
H1 pilot A — the budget's predictions, written before the grid runs.

For every design point in `pilot_a_grid.py`, builds the first-layer weight
matrix exactly as `r3_common.run_spec` will (fixed fan-in, then
`perturb_weights`) for each of the 5 seeds, and records its effective fan-in:
the participation ratio `PR = (sum w)^2 / sum w^2` averaged over neurons with
any input, the fraction of neurons with none, and the coefficient of variation
of each neuron's summed weight (PA4's spread).

It then predicts each record's ceiling from that record's B2 fit,
`1/R = c * (PR_exact / PR_level) / r_0 + 1/Lambda_rest` in `S/N` terms, with c
and Lambda_rest solved from B2's 20 and 40 Hz cells. No parameter is fitted to
pilot A.

Outputs, in `results/h1_hardware_cost/analysis/`:
  pilot_a_predictions.csv          one row per design point and record
  pilot_a_prediction_summary.csv   one row per design point, medians over records

Usage:
    uv run python scripts/h1_hardware_cost/pilot_a_predictions.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from neuroflux_core.metrics.cascade import excess_to_affine_db, linear_to_db

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_b2 as b2  # noqa: E402
import analyse_r3 as r3  # noqa: E402
import pilot_a_grid as grid  # noqa: E402
import pilot_b_cost as pilot  # noqa: E402
from r3_common import fixed_fan_in_weights, perturb_weights  # noqa: E402

SEEDS = 5

Record = dict[str, Any]


def matrix(n_in: int, seed: int, level: grid.Level) -> np.ndarray:
    """The first-layer weights `run_spec` builds for this design point."""
    weights = fixed_fan_in_weights(
        n_in,
        grid.LAYER_WIDTH,
        grid.FAN_IN,
        np.random.default_rng(np.random.SeedSequence([seed, 30_000])),
    )
    bits, sigma, stuck = level
    if bits is None and sigma is None and stuck is None:
        return weights
    return perturb_weights(weights, seed, 0, bits=bits, sigma=sigma, stuck=stuck)


def fan_in_stats(n_in: int, level: grid.Level) -> dict[str, float]:
    prs, dead, spread = [], [], []
    for seed in range(SEEDS):
        w = matrix(n_in, seed, level).astype(np.float64)
        total = w.sum(axis=0)
        square = (w**2).sum(axis=0)
        live = square > 0.0
        prs.append(float(np.mean(total[live] ** 2 / square[live])))
        dead.append(float(np.mean(~live)))
        spread.append(float(np.std(total[live]) / np.mean(total[live])))
    return {
        "pr": float(np.mean(prs)),
        "dead_fraction": float(np.mean(dead)),
        "summed_weight_cv": float(np.mean(spread)),
    }


def b2_fits(n_in: int) -> dict[str, tuple[float, float]]:
    """Record -> (c, 1/Lambda_rest) from B2's 20 and 40 Hz cells."""
    cells = b2.load_cells()
    at = {hz: b2.ceilings(cells, (n_in, grid.LAYER_WIDTH, hz)) for hz in (20.0, 40.0)}
    fits: dict[str, tuple[float, float]] = {}
    for unit in sorted(set(at[20.0]) & set(at[40.0])):
        y20 = b2.inverse_excess(at[20.0][unit])
        y40 = b2.inverse_excess(at[40.0][unit])
        c = (y20 - y40) / (1.0 / 20.0 - 1.0 / 40.0)
        fits[unit] = (c, y40 - c / 40.0)
    return fits


def predict(c: float, inv_rest: float, pr_ratio: float, encoder_hz: float) -> float:
    inverse = c * pr_ratio / encoder_hz + inv_rest
    return excess_to_affine_db(linear_to_db(1.0 / inverse)) if inverse > 0 else math.inf


def main() -> None:
    r3.OUT = pilot.ANALYSIS
    rows: list[Record] = []
    summary: list[Record] = []
    for n_in in grid.INPUT_WIDTHS:
        fits = b2_fits(n_in)
        stats = {
            name: fan_in_stats(n_in, level) for name, level in grid.levels().items()
        }
        exact_pr = stats["exact"]["pr"]
        for name in grid.levels():
            ratio = exact_pr / stats[name]["pr"]
            print(
                f"  [{n_in},{grid.LAYER_WIDTH}] {name:<10} PR {stats[name]['pr']:5.2f} "
                f"(x{ratio:.2f} counting noise), no input "
                f"{stats[name]['dead_fraction']:.3f}, summed-weight CV "
                f"{stats[name]['summed_weight_cv']:.2f}"
            )
            for encoder_hz in grid.ENCODER_RATES_HZ:
                changes = []
                for unit, (c, inv_rest) in fits.items():
                    exact = predict(c, inv_rest, 1.0, encoder_hz)
                    level = predict(c, inv_rest, ratio, encoder_hz)
                    changes.append(level - exact)
                    rows.append(
                        {
                            "n_in": n_in,
                            "n_1": grid.LAYER_WIDTH,
                            "level": name,
                            "encoder_hz": encoder_hz,
                            "unit": unit,
                            "pr_ratio": ratio,
                            "predicted_exact_db": exact,
                            "predicted_level_db": level,
                            "predicted_change_db": level - exact,
                        }
                    )
                summary.append(
                    {
                        "n_in": n_in,
                        "n_1": grid.LAYER_WIDTH,
                        "level": name,
                        "encoder_hz": encoder_hz,
                        **stats[name],
                        "pr_ratio": ratio,
                        "median_predicted_change_db": float(np.median(changes)),
                    }
                )
    r3.write("pilot_a_predictions.csv", rows)
    r3.write("pilot_a_prediction_summary.csv", summary)


if __name__ == "__main__":
    main()
