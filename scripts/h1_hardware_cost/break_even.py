"""
H1 break-even — which design is cheapest, as a function of what a spike costs.

An event-driven chip spends energy per synaptic operation and per routed spike.
Neither figure is measured in this project, and the paper must not assume one.
Writing a design's event cost in synaptic-operation equivalents,

    cost(rho) = synops_per_s + rho * spikes_per_s,    rho = E_spike / E_synop,

leaves the one unknown as a ratio a reader can supply for their own chip. This
script is descriptive: it reuses committed counts and medians and runs no test.

  break_even_designs.csv  the confirmation grid's six designs on MIT-BIH
                          (noiseless input): their event costs, and every
                          pairwise crossover ratio rho* at which two designs
                          cost the same
  break_even_targets.csv  for each fidelity target and each rho on a log grid,
                          the cheapest measured MIT-BIH design (pilot B's R3/R4b
                          points and B2, exact weights) whose median ceiling
                          reaches the target. The confirmation designs are left
                          out: they were scored on other ECG windows (seeds
                          10-19), whose ceilings run higher, so their medians
                          are not comparable with the earlier grids'.
  break_even_switches.csv for each target, the exact rho at which the cheapest
                          design changes, from the two designs' counts rather
                          than the log grid

Usage:
    uv run python scripts/h1_hardware_cost/break_even.py
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_r3 as r3  # noqa: E402
import pilot_b_cost as pilot  # noqa: E402

TARGETS_DB: tuple[float, ...] = (3.0, 4.0, 5.0, 5.5)
RHOS: tuple[float, ...] = tuple(float(x) for x in np.logspace(-2, 2, 41))

Record = dict[str, Any]


def read(name: str) -> list[dict[str, str]]:
    with (pilot.ANALYSIS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def confirmation_designs() -> list[Record]:
    return [
        {
            "design": r["design"],
            "ceiling_db": float(r["ceiling_db"]),
            "synops_per_s": float(r["synops_per_s"]),
            "spikes_per_s": float(r["spikes_per_s"]),
        }
        for r in read("confirm_cells.csv")
        if r["signal"] == "mit-bih" and r["input_snr"] == "inf"
    ]


def landscape() -> list[Record]:
    points = [
        {
            "design": r["design"],
            "source": "R3/R4b",
            "ceiling_db": float(r["ceiling_db"]),
            "synops_per_s": float(r["synops_per_s"]),
            "spikes_per_s": float(r["spikes_per_s"]),
        }
        for r in read("pilot_b_points.csv")
        if r["signal"] == "mit-bih"
    ]
    points += [
        {
            "design": r["design"],
            "source": "B2",
            "ceiling_db": float(r["ceiling_db"]),
            "synops_per_s": float(r["synops_per_s"]),
            "spikes_per_s": float(r["spikes_per_s"]),
        }
        for r in read("b2_cells.csv")
    ]
    return points


def crossovers(designs: list[Record]) -> list[Record]:
    rows: list[Record] = []
    for a, b in itertools.combinations(designs, 2):
        d_synops = b["synops_per_s"] - a["synops_per_s"]
        d_spikes = b["spikes_per_s"] - a["spikes_per_s"]
        # a + rho*sa = b + rho*sb  ->  rho* = -d_synops / d_spikes
        rho = -d_synops / d_spikes if d_spikes != 0 else math.nan
        if rho > 0:
            cheaper_below = a["design"] if d_spikes < 0 else b["design"]
            relation = f"{cheaper_below} is cheaper when rho < {rho:.3g}"
        else:
            cheaper = a if a["synops_per_s"] <= b["synops_per_s"] else b
            relation = f"{cheaper['design']} is cheaper at every rho"
        rows.append(
            {
                "design_a": a["design"],
                "design_b": b["design"],
                "ceiling_a_db": a["ceiling_db"],
                "ceiling_b_db": b["ceiling_db"],
                "crossover_rho": rho if rho > 0 else math.nan,
                "relation": relation,
            }
        )
    return rows


def cheapest(points: list[Record]) -> list[Record]:
    rows: list[Record] = []
    for target in TARGETS_DB:
        eligible = [p for p in points if p["ceiling_db"] >= target]
        for rho in RHOS:
            if not eligible:
                continue
            best = min(
                eligible, key=lambda p: p["synops_per_s"] + rho * p["spikes_per_s"]
            )
            rows.append(
                {
                    "target_db": target,
                    "rho": rho,
                    "design": best["design"],
                    "source": best["source"],
                    "ceiling_db": best["ceiling_db"],
                    "cost_synop_equivalents_per_s": best["synops_per_s"]
                    + rho * best["spikes_per_s"],
                    "n_eligible": len(eligible),
                }
            )
    return rows


def main() -> None:
    r3.OUT = pilot.ANALYSIS
    designs = confirmation_designs()
    rows = crossovers(designs)
    print("Confirmation designs, MIT-BIH, noiseless:")
    for d in designs:
        print(
            f"  {d['design']}: {d['ceiling_db']:.2f} dB, "
            f"{d['synops_per_s']:.0f} synops/s, {d['spikes_per_s']:.0f} spikes/s"
        )
    for r in rows:
        print(f"  {r['design_a']} vs {r['design_b']}: {r['relation']}")
    r3.write("break_even_designs.csv", [{**d} for d in designs] + rows)

    table = cheapest(landscape())
    print("\nCheapest measured design reaching each target:")
    for target in TARGETS_DB:
        runs: list[str] = []
        previous = None
        for r in (t for t in table if t["target_db"] == target):
            if r["design"] != previous:
                runs.append(
                    f"rho >= {r['rho']:.3g}: {r['design']} ({r['ceiling_db']:.2f} dB)"
                )
                previous = r["design"]
        print(f"  >= {target:g} dB -> " + "; ".join(runs))
    r3.write("break_even_targets.csv", table)

    switches: list[Record] = []
    by_design = {p["design"]: p for p in landscape()}
    for target in TARGETS_DB:
        winners: list[str] = []
        for r in (t for t in table if t["target_db"] == target):
            if not winners or winners[-1] != r["design"]:
                winners.append(r["design"])
        for low, high in itertools.pairwise(winners):
            a, b = by_design[low], by_design[high]
            rho = (b["synops_per_s"] - a["synops_per_s"]) / (
                a["spikes_per_s"] - b["spikes_per_s"]
            )
            switches.append(
                {
                    "target_db": target,
                    "cheapest_below": low,
                    "cheapest_above": high,
                    "switch_rho": rho,
                }
            )
            print(f"  >= {target:g} dB: {low} -> {high} at rho = {rho:.3g}")
    r3.write("break_even_switches.csv", switches)


if __name__ == "__main__":
    main()
