"""
H1 confirmation grid analysis — the tests `confirm_grid.py` states.

Every rule is in `confirm_grid.py`'s docstring, committed before the grid ran.
Statistics are R3's: records are units on MIT-BIH and seeds on the synthetic
signals, SNR averages a unit's `ok` runs, paired tests pair units, TOST at
+/-0.5 dB, BCa intervals. Costs come from each run's measured rates, as in
pilot B (`pilot_b_cost.run_costs`).

Outputs, in `results/h1_hardware_cost/analysis/`:
  confirm_cells.csv   one row per signal x input x design: output SNR (the
                      ceiling at noiseless input), measured event costs, and the
                      design's static costs: weight memory, counted at 32 bits
                      for exact weights with an input address per synapse of a
                      fixed-fan-in layer (dense layers need none), and the
                      synapse reads and neuron updates a clock-driven chip
                      makes at 250 steps per second
  confirm_tests.csv   C1-C5, G1

Usage:
    uv run python scripts/h1_hardware_cost/analyse_confirm.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_r3 as r3  # noqa: E402
import confirm_grid as grid  # noqa: E402
import pilot_b_cost as pilot  # noqa: E402
from r3_common import DT  # noqa: E402

NON_INFERIORITY_DB = -0.5
NOISELESS = "inf"

Row = dict[str, str]
Record = dict[str, Any]
Cell = tuple[str, str, str]  # (signal, input SNR as written, design)


def design_of(row: Row) -> str:
    fields = dict(part.split("=", 1) for part in row["spec_key"].split("|"))
    layers = tuple(int(w) for w in row["layers"].split("-"))
    fan_in = int(float(row["fan_in"])) or None
    bits = int(fields["weight_bits"]) if "weight_bits" in fields else None
    encoder_hz = float(fields["encoder_rate_hz"])
    for name, (d_layers, _, d_encoder, d_fan_in, d_bits) in grid.DESIGNS.items():
        if (d_layers, d_encoder, d_fan_in, d_bits) == (
            layers,
            encoder_hz,
            fan_in,
            bits,
        ):
            return name
    raise ValueError(f"No design matches {row['spec_key']}")


def load_cells() -> dict[Cell, list[Row]]:
    path = pilot.source(grid.OUTPUT)
    if path is None:
        raise SystemExit(f"{grid.OUTPUT} not found; run confirm_grid.py first.")
    cells: dict[Cell, list[Row]] = defaultdict(list)
    for row in r3.load(path):
        cells[(row["signal"], row["input_snr_nominal"], design_of(row))].append(row)
    return cells


def ceilings(cells: dict[Cell, list[Row]], cell: Cell) -> dict[str, float]:
    values: dict[str, float] = r3.unit_values(cells[cell], "snr_affine_db")
    return values


def costs(cells: dict[Cell, list[Row]], cell: Cell) -> dict[str, float]:
    """Median over units of each unit's mean cost over its `ok` runs."""
    signal, _, name = cell
    layers, rates, _, fan_in, _ = grid.DESIGNS[name]
    design = pilot.Design(
        signal=signal,
        operating_point="controlled",
        layers="-".join(str(w) for w in layers),
        fan_in=fan_in or 0,
        rate_targets=";".join(f"{r:g}" for r in rates),
        epochs=0,
        stdp_tau=20.0,
    )
    per_unit: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in cells[cell]:
        if row["verdict"] != "ok":
            continue
        for metric, value in pilot.run_costs(design, row).items():
            per_unit[r3.unit_id(row)][metric].append(value)
    metrics = ("synapses", "synops_per_s", "spikes_per_s")
    return {
        m: r3.median(float(np.mean(v[m])) for v in per_unit.values()) for m in metrics
    }


EXACT_WEIGHT_BITS = 32


def static_costs(name: str) -> dict[str, float]:
    """Weight memory and clock-driven work, from the design alone."""
    layers, _, _, fan_in, bits = grid.DESIGNS[name]
    steps_per_s = 1000.0 / DT
    weight_bits = bits if bits is not None else EXACT_WEIGHT_BITS
    memory = 0
    synapses = 0
    for k in range(1, len(layers)):
        sparse = k == 1 and fan_in is not None
        count = (fan_in if sparse else layers[k - 1]) * layers[k]  # type: ignore[operator]
        synapses += count
        address = math.ceil(math.log2(layers[k - 1])) if sparse else 0
        memory += count * (weight_bits + address)
    return {
        "weight_memory_bits": float(memory),
        "clock_synapse_reads_per_s": synapses * steps_per_s,
        "clock_neuron_updates_per_s": sum(layers[1:]) * steps_per_s,
    }


def compare(
    cells: dict[Cell, list[Row]], signal: str, snr: str, reference: str, other: str
) -> Record:
    record: Record = r3.compare(
        ceilings(cells, (signal, snr, reference)),
        ceilings(cells, (signal, snr, other)),
        r3.SNR_MARGIN,
    )
    record["comparison"] = f"{signal}, input {snr}: {other} minus {reference}"
    return record


def show(record: Record, verdict: str) -> None:
    print(
        f"  {record['test']}: {r3.fmt(record['median_diff'])} dB "
        f"[{r3.fmt(record['ci_low'])}, {r3.fmt(record['ci_high'])}], "
        f"TOST p {record['tost_p']:.2g}, n {record['n_units']} -> {verdict}"
    )


def main() -> None:
    r3.OUT = pilot.ANALYSIS
    cells = load_cells()
    runs = sum(len(v) for v in cells.values())
    ok = sum(r["verdict"] == "ok" for v in cells.values() for r in v)
    print(f"H1 confirmation grid: {runs} runs in {len(cells)} cells, {ok} ok\n")
    tests: list[Record] = []

    c1 = compare(cells, "mit-bih", NOISELESS, "D1", "D2")
    c1.update(test="C1 headline, D2 vs D1", verdict=_held(c1["equivalent"]))
    c2 = compare(cells, "mit-bih", NOISELESS, "D2", "D3")
    c2.update(test="C2 two bits, D3 vs D2", verdict=_held(c2["equivalent"]))
    c3 = compare(cells, "mit-bih", NOISELESS, "D5", "D4")
    c3.update(
        test="C3 break-even pair, D4 vs D5 (non-inferiority)",
        verdict=_held(c3["ci_low"] > NON_INFERIORITY_DB),
    )
    for record in (c1, c2, c3):
        show(record, record["verdict"])
        tests.append(record)

    cost = {d: costs(cells, ("mit-bih", NOISELESS, d)) for d in grid.DESIGNS}
    d2_ratio = cost["D2"]["synops_per_s"] / cost["D1"]["synops_per_s"]
    d4_ratio = cost["D4"]["synops_per_s"] / cost["D5"]["synops_per_s"]
    extra_spikes = cost["D4"]["spikes_per_s"] - cost["D5"]["spikes_per_s"]
    break_even = (
        (cost["D5"]["synops_per_s"] - cost["D4"]["synops_per_s"]) / extra_spikes
        if extra_spikes > 0
        else math.nan
    )
    c4_held = (
        d2_ratio <= 0.13
        and cost["D2"]["spikes_per_s"] < cost["D1"]["spikes_per_s"]
        and d4_ratio <= 0.26
        and extra_spikes > 0
    )
    c4: Record = {
        "test": "C4 costs",
        "d2_synops_ratio_to_d1": d2_ratio,
        "d2_spikes_ratio_to_d1": cost["D2"]["spikes_per_s"]
        / cost["D1"]["spikes_per_s"],
        "d4_synops_ratio_to_d5": d4_ratio,
        "d4_spikes_ratio_to_d5": cost["D4"]["spikes_per_s"]
        / cost["D5"]["spikes_per_s"],
        "break_even_spike_to_synop_energy": break_even,
        "verdict": _held(c4_held),
    }
    print(
        f"  C4 costs: D2 synops x{d2_ratio:.3f} and spikes "
        f"x{c4['d2_spikes_ratio_to_d1']:.2f} of D1; D4 synops x{d4_ratio:.3f} and "
        f"spikes x{c4['d4_spikes_ratio_to_d5']:.2f} of D5; break-even spike energy "
        f"{break_even:.2f} synaptic operations -> {c4['verdict']}"
    )
    tests.append(c4)

    c5 = [compare(cells, "mit-bih", snr, "D1", "D2") for snr in ("10.0", "0.0")]
    c5_held = all(r["equivalent"] for r in c5)
    for record, snr in zip(c5, ("10 dB", "0 dB"), strict=True):
        record.update(test=f"C5 D2 vs D1 at {snr} input", verdict=_held(c5_held))
        show(record, "equivalent" if record["equivalent"] else "not equivalent")
        tests.append(record)
    print(f"  C5 -> {_held(c5_held)}")

    g1 = [compare(cells, s, NOISELESS, "D1", "D2") for s in grid.SYNTHETIC_SIGNALS]
    g1_held = all(r["ci_low"] > NON_INFERIORITY_DB for r in g1)
    for record, signal in zip(g1, grid.SYNTHETIC_SIGNALS, strict=True):
        record.update(test=f"G1 D2 vs D1 on {signal}", verdict=_held(g1_held))
        show(
            record,
            "non-inferior" if record["ci_low"] > NON_INFERIORITY_DB else "not",
        )
        tests.append(record)
    print(f"  G1 -> {_held(g1_held)}")
    r3.write("confirm_tests.csv", tests)

    rows: list[Record] = []
    print("\nG2 — ceilings (median over units, dB) and costs:")
    for signal in ("mit-bih", *grid.SYNTHETIC_SIGNALS):
        for snr in (NOISELESS, "10.0", "0.0"):
            line = []
            for name in grid.DESIGNS:
                cell = (signal, snr, name)
                value = r3.median(ceilings(cells, cell).values())
                cell_cost = costs(cells, cell)
                ok_frac = float(np.mean([r["verdict"] == "ok" for r in cells[cell]]))
                rows.append(
                    {
                        "signal": signal,
                        "input_snr": snr,
                        "design": name,
                        "ceiling_db": value,
                        **cell_cost,
                        **static_costs(name),
                        "ok_fraction": ok_frac,
                        "n_units": len(ceilings(cells, cell)),
                    }
                )
                line.append(f"{name} {value:.2f}")
            print(f"  {signal:<8} input {snr:>4}: " + ", ".join(line))
    r3.write("confirm_cells.csv", rows)


def _held(condition: bool) -> str:
    return "held" if condition else "did not hold"


if __name__ == "__main__":
    main()
