"""
H1 pilot B — what each measured design point costs in hardware terms.

Every R3 and R4b grid records each run's encoder rate and every LIF layer's
rate, so the synaptic work a design point does can be computed from committed
rows without simulating anything. This pilot puts that cost next to the
noiseless-input ceiling (causal, affine-corrected, as everywhere since R3) for
every design point measured on MIT-BIH, and on the other signals where a grid
exists.

Cost model, per second of signal, for a feed-forward network with widths
N_0 (inputs) ... N_L and per-neuron mean rates r_0 (encoder) ... r_L:

  synapses       S = sum_k S_k, with S_1 = K * N_1 at a fixed fan-in K (R4b,
                 first weight matrix only) and S_k = N_{k-1} * N_k otherwise.
  synaptic ops   sum_k r_{k-1} * S_k: one weight read per presynaptic spike per
                 synapse, the unit an event-driven chip does work in. Exact in
                 expectation here, because every dense layer has uniform
                 fan-out and the tiled input gives every input channel the same
                 expected rate.
  spikes         sum_k N_k * r_k including the encoder: address-event traffic.
  LIF neurons    sum_{k>=1} N_k; a clock-driven chip updates each one every
                 4 ms step and touches every synapse every step, so for it the
                 cost is S and the neuron count, and firing rates do not matter.

What this pilot cannot see. Every rate-controlled grid holds the encoder at
40 Hz, and the input layer carries most of the synaptic work, so the encoder
rate — probably the largest cost lever — is unmeasured. Weights that STDP
drives to zero are counted as synapses; a trained network's structural count
is an upper bound.

Question and reading rule. The ceilings were published in R3 and R4b before
this pilot was written; only the cost side is new, so the rule below is a
framing judgement, not a pre-registered test. Cost-against-fidelity becomes
the MCSoC paper's frame if, on MIT-BIH:

  1. cost changes the design choice: a design point on the synaptic-operation
     front reaches within 0.5 dB (the project's equivalence margin) of the best
     ceiling for at most a quarter of that point's synaptic operations, the
     distance measured as the paired median difference over records, as every
     R3 comparison is; and
  2. the levers separate: design points from at least two levers (fan-in,
     input width, output rate, depth) lie on that front.

If either fails, cost rises with fidelity along one lever and the frame adds a
sentence to what the ICEE paper already says about fan-in.

The front treats costs within 5% as equal when deciding dominance: calibration
leaves measured rates about 1% off target, which would otherwise let a design
point onto the front by being 0.2% cheaper.

Paired contrasts pair records, as in R3: median difference, BCa interval,
Wilcoxon p. They are descriptive; no family is Holm-adjusted.

Usage:
    uv run python scripts/h1_hardware_cost/pilot_b_cost.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
import analyse_r3 as r3  # noqa: E402
from figures_r3 import (  # noqa: E402
    BLUE,
    GREEN,
    IEEE_RC,
    INK,
    REFERENCE,
    SINGLE_COLUMN_IN,
    VERMILLION,
    WINE,
)
from r3_common import split_floats  # noqa: E402

RESULTS = Path("results/h1_hardware_cost")
ANALYSIS = RESULTS / "analysis"
FIGURES = RESULTS / "figures"
R3_RESULTS = Path("results/r3_characterisation")
R4B_RESULTS = Path("results/r4b_mechanism")

# Grids in priority order: a design point measured by more than one grid is
# taken from the first that has it, so identical runs are not counted twice.
GRIDS: tuple[tuple[str, Path], ...] = (
    ("r4b-fan-in", R4B_RESULTS / "fan_in.csv"),
    ("r4b-width", R4B_RESULTS / "width.csv"),
    ("c3-rate", R3_RESULTS / "c3_rate.csv"),
    ("c3-rate-signals", R3_RESULTS / "c3_rate_signals.csv"),
    ("c2", R3_RESULTS / "c2_invariances.csv"),
)
NEAR_BEST_DB = 0.5
COST_TIE = 0.05
COST_FRACTION = 0.25

Row = dict[str, str]
Record = dict[str, Any]


@dataclass(frozen=True)
class Design:
    signal: str
    operating_point: str
    layers: str
    fan_in: int
    rate_targets: str
    epochs: int
    stdp_tau: float

    @property
    def widths(self) -> list[int]:
        return [int(w) for w in self.layers.split("-")]

    def label(self) -> str:
        k = "" if not self.fan_in else f" K={self.fan_in}"
        rates = self.rate_targets or "shipped"
        trained = f" {self.epochs} ep" if self.epochs else ""
        if self.epochs and self.stdp_tau != 20.0:
            trained += f" tau={self.stdp_tau:g}"
        return f"[{self.layers.replace('-', ',')}]{k} @{rates}{trained}"

    def lever(self) -> str:
        """The family a design point belongs to, for the figure and the front."""
        widths = self.widths
        if self.operating_point == "shipped":
            return "shipped"
        if self.epochs:
            return "training"
        if len(widths) == 2 and widths[0] == widths[1] and widths[0] >= 128:
            return "fan-in" if self.fan_in else "one layer, dense"
        if len(widths) != 3:
            return "depth"
        if widths[0] != 32:
            return "input width"
        if self.layers == "32-16-8" and self.rate_targets == "40;40":
            return "reference"
        if widths[2] != 8 and split_floats(self.rate_targets) == [40.0, 40.0]:
            return "output width"
        return "output rate"


def source(path: Path) -> Path | None:
    packed = path.with_name(path.name + ".gz")
    if path.exists():
        return path
    return packed if packed.exists() else None


def design_of(row: Row) -> Design | None:
    """The design point a noiseless, 20 ms-decoder run measures, else None."""
    if row["input_snr_nominal"] != "inf" or row["decoder_tau"] != "20.0":
        return None
    return Design(
        signal=row["signal"],
        operating_point=row["operating_point"],
        layers=row["layers"],
        fan_in=int(float(row.get("fan_in") or 0)),
        rate_targets=row.get("layer_rate_targets_hz", ""),
        epochs=int(row["epochs"]),
        stdp_tau=float(row["stdp_tau"]),
    )


def run_costs(design: Design, row: Row) -> dict[str, float]:
    widths = design.widths
    rates = [r3.fnum(row["encoder_rate_hz"]), *split_floats(row["layer_rates_hz"])]
    if len(rates) != len(widths):
        raise ValueError(f"{design.label()}: {len(rates)} rates, {len(widths)} layers")
    synapses = [
        (design.fan_in if k == 1 and design.fan_in else widths[k - 1]) * widths[k]
        for k in range(1, len(widths))
    ]
    return {
        "synapses": float(sum(synapses)),
        "synops_per_s": float(
            sum(rates[k - 1] * synapses[k - 1] for k in range(1, len(widths)))
        ),
        "spikes_per_s": float(sum(n * r for n, r in zip(widths, rates, strict=True))),
        "lif_neurons": float(sum(widths[1:])),
        "input_synops_share": float(
            rates[0]
            * synapses[0]
            / sum(rates[k - 1] * synapses[k - 1] for k in range(1, len(widths)))
        ),
    }


def collect() -> dict[Design, dict[str, dict[str, list[float]]]]:
    """Design -> unit -> metric -> per-run values, `ok` runs only."""
    points: dict[Design, dict[str, dict[str, list[float]]]] = {}
    owner: dict[Design, str] = {}
    for grid, path in GRIDS:
        found = source(path)
        if found is None:
            print(f"  {path} not found; skipped")
            continue
        taken: dict[Design, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in r3.load(found):
            design = design_of(row)
            if design is None or owner.get(design, grid) != grid:
                continue
            owner[design] = grid
            unit = taken[design][r3.unit_id(row)]
            unit["runs"].append(1.0)
            if row["verdict"] != "ok":
                continue
            unit["ok"].append(1.0)
            unit["ceiling_db"].append(r3.fnum(row["snr_affine_db"]))
            for name, value in run_costs(design, row).items():
                unit[name].append(value)
        points.update(taken)
        print(f"  {grid}: {len(taken)} design points from {found}")
    return points


def unit_means(
    units: dict[str, dict[str, list[float]]], metric: str
) -> dict[str, float]:
    return {
        unit: float(np.mean(values[metric]))
        for unit, values in units.items()
        if values[metric]
    }


def summarise(
    points: dict[Design, dict[str, dict[str, list[float]]]],
) -> list[Record]:
    records: list[Record] = []
    for design, units in points.items():
        runs = sum(len(v["runs"]) for v in units.values())
        ok = sum(len(v["ok"]) for v in units.values())
        record: Record = {
            "signal": design.signal,
            "design": design.label(),
            "lever": design.lever(),
            "operating_point": design.operating_point,
            "layers": design.layers,
            "fan_in": design.fan_in or "dense",
            "rate_targets_hz": design.rate_targets,
            "epochs": design.epochs,
            "stdp_tau": design.stdp_tau,
            "n_units": len(unit_means(units, "ceiling_db")),
            "ok_fraction": ok / runs if runs else math.nan,
        }
        for metric in (
            "ceiling_db",
            "synapses",
            "synops_per_s",
            "spikes_per_s",
            "lif_neurons",
            "input_synops_share",
        ):
            record[metric] = r3.median(unit_means(units, metric).values())
        records.append(record)
    records.sort(key=lambda r: (r["signal"], r["synops_per_s"], -r["ceiling_db"]))
    return records


def pareto(records: list[Record], cost: str) -> list[Record]:
    """
    Points no other point beats on both ceiling and `cost`.

    Costs within `COST_TIE` of each other count as equal, so a point is
    dominated by one with a higher ceiling at a cost no more than 5% above its
    own, or by one at least 5% cheaper with a ceiling no lower.
    """
    finite = [r for r in records if math.isfinite(r["ceiling_db"])]

    def dominates(o: Record, r: Record) -> bool:
        higher = o["ceiling_db"] > r["ceiling_db"]
        no_lower = o["ceiling_db"] >= r["ceiling_db"]
        return bool(
            (higher and o[cost] <= r[cost] * (1 + COST_TIE))
            or (no_lower and o[cost] * (1 + COST_TIE) < r[cost])
        )

    front = [r for r in finite if not any(dominates(o, r) for o in finite)]
    return sorted(front, key=lambda r: r[cost])


# --------------------------------------------------------------------------
# Paired contrasts on MIT-BIH
# --------------------------------------------------------------------------


def contrast(
    points: dict[Design, dict[str, dict[str, list[float]]]],
    name: str,
    reference: Design,
    other: Design,
) -> Record | None:
    if reference not in points or other not in points:
        print(f"  {name}: design point missing; skipped")
        return None
    ref_units, other_units = points[reference], points[other]
    record: Record = r3.compare(
        unit_means(ref_units, "ceiling_db"),
        unit_means(other_units, "ceiling_db"),
        r3.SNR_MARGIN,
    )
    ratios: dict[str, float] = {}
    for metric in ("synapses", "synops_per_s", "spikes_per_s", "lif_neurons"):
        ref = unit_means(ref_units, metric)
        oth = unit_means(other_units, metric)
        shared = ref.keys() & oth.keys()
        ratios[metric] = r3.median(oth[u] / ref[u] for u in shared if ref[u] > 0)
    record.update(
        {
            "contrast": name,
            "reference": reference.label(),
            "other": other.label(),
            **{f"{m}_ratio": v for m, v in ratios.items()},
            "db_per_synops_doubling": (
                record["median_diff"] / math.log2(ratios["synops_per_s"])
                if ratios["synops_per_s"] > 0
                and abs(math.log2(ratios["synops_per_s"])) >= math.log2(1 + COST_TIE)
                else math.nan
            ),
        }
    )
    return record


def mit(
    layers: str,
    rates: str,
    *,
    fan_in: int = 0,
    epochs: int = 0,
    operating_point: str = "controlled",
) -> Design:
    return Design("mit-bih", operating_point, layers, fan_in, rates, epochs, 20.0)


def contrasts(points: dict[Design, dict[str, dict[str, list[float]]]]) -> list[Record]:
    shipped = mit("32-16-8", "40;40")
    wanted: list[tuple[str, Design, Design]] = [
        ("output rate 5 Hz vs 40 Hz", shipped, mit("32-16-8", "40;5")),
        ("every layer 10 Hz vs 40 Hz", shipped, mit("32-16-8", "10;10")),
        ("output width 64 vs 8", shipped, mit("32-16-64", "40;40")),
        ("input width 256 vs 32, output 40 Hz", shipped, mit("256-16-8", "40;40")),
        (
            "input width 256 vs 32, output 5 Hz",
            mit("32-16-8", "40;5"),
            mit("256-16-8", "40;5"),
        ),
        ("depth 2 vs 1 LIF layer", mit("32-8", "40"), shipped),
        ("depth 3 vs 2 LIF layers", shipped, mit("32-16-16-8", "40;40;40")),
        (
            "10 STDP epochs vs none, held rate",
            shipped,
            mit("32-16-8", "40;40", epochs=10),
        ),
        (
            "[128,128] K=8 vs dense, 20 Hz",
            mit("128-128", "20"),
            mit("128-128", "20", fan_in=8),
        ),
        (
            "[512,512] K=16 vs dense, 20 Hz",
            mit("512-512", "20"),
            mit("512-512", "20", fan_in=16),
        ),
        (
            "[512,512] K=16 vs [128,128] K=8, 20 Hz",
            mit("128-128", "20", fan_in=8),
            mit("512-512", "20", fan_in=16),
        ),
        (
            "[128,128] K=8 at 20 Hz vs shipped architecture",
            shipped,
            mit("128-128", "20", fan_in=8),
        ),
        (
            "[512,512] K=16 at 20 Hz vs shipped architecture",
            shipped,
            mit("512-512", "20", fan_in=16),
        ),
        (
            "rate-controlled vs shipped operating point",
            mit("32-16-8", "", operating_point="shipped"),
            shipped,
        ),
    ]
    records = []
    for name, reference, other in wanted:
        record = contrast(points, name, reference, other)
        if record is not None:
            records.append(record)
    return records


# --------------------------------------------------------------------------
# Reading rule
# --------------------------------------------------------------------------


def reading(
    points: dict[Design, dict[str, dict[str, list[float]]]],
    front: list[Record],
    mit_records: list[Record],
) -> list[str]:
    by_label = {
        d.label(): units for d, units in points.items() if d.signal == "mit-bih"
    }
    best = max(mit_records, key=lambda r: r["ceiling_db"])
    best_units = unit_means(by_label[best["design"]], "ceiling_db")
    lines = [
        f"Best MIT-BIH ceiling: {best['design']} {best['ceiling_db']:.2f} dB at "
        f"{best['synops_per_s']:.3g} synaptic ops/s"
    ]
    near = []
    for r in front:
        if r["synops_per_s"] > COST_FRACTION * best["synops_per_s"]:
            continue
        gap = r3.compare(
            unit_means(by_label[r["design"]], "ceiling_db"), best_units, r3.SNR_MARGIN
        )
        lines.append(
            f"  {r['design']} at {r['synops_per_s'] / best['synops_per_s']:.3g}x "
            f"the synaptic ops: best minus it {r3.fmt(gap['median_diff'])} dB paired "
            f"[{r3.fmt(gap['ci_low'])}, {r3.fmt(gap['ci_high'])}], "
            f"{best['ceiling_db'] - r['ceiling_db']:.2f} dB between medians"
        )
        if gap["median_diff"] <= NEAR_BEST_DB:
            near.append(r)
    levers = sorted({r["lever"] for r in front})
    lines += [
        f"Criterion 1 (a front point within {NEAR_BEST_DB} dB, paired, at <= "
        f"{COST_FRACTION:g}x the best point's synaptic ops): "
        + ("MET by " + "; ".join(r["design"] for r in near) if near else "NOT met"),
        f"Criterion 2 (at least two levers on the front): levers {levers} -> "
        + ("MET" if len(set(levers) - {"shipped", "training"}) >= 2 else "NOT met"),
    ]
    return lines


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

LEVER_STYLE: dict[str, tuple[str, str]] = {
    "fan-in": (GREEN, "D"),
    "one layer, dense": (GREEN, "d"),
    "input width": (BLUE, "o"),
    "output rate": (VERMILLION, "s"),
    "output width": (VERMILLION, "x"),
    "depth": (WINE, "^"),
    "training": (REFERENCE, "v"),
    "reference": (INK, "o"),
    "shipped": (INK, "*"),
}
# Design points named on the synaptic-operation panel.
ANNOTATE: dict[str, tuple[str, tuple[float, float]]] = {
    "[32,8] @40": ("[32,8]", (4, 4)),
    "[128,128] K=8 @20": ("[128,128], K=8", (-4, 5)),
    "[512,512] K=16 @20": ("[512,512], K=16", (-4, 5)),
    "[32,16,8] @40;40": ("[32,16,8], 40 Hz", (5, -9)),
}


def draw(ax: Axes, records: list[Record], cost: str, xlabel: str) -> None:
    for record in records:
        colour, marker = LEVER_STYLE[record["lever"]]
        size = 22 if marker == "*" else 11
        if marker == "x":
            ax.scatter(
                record[cost],
                record["ceiling_db"],
                s=size,
                marker=marker,
                color=colour,
                linewidths=0.7,
                zorder=3,
            )
        else:
            ax.scatter(
                record[cost],
                record["ceiling_db"],
                s=size,
                marker=marker,
                facecolors=colour,
                edgecolors="white",
                linewidths=0.3,
                zorder=4 if record["lever"] == "reference" else 3,
            )
        if cost == "synops_per_s" and record["design"] in ANNOTATE:
            text, offset = ANNOTATE[record["design"]]
            ax.annotate(
                text,
                (record[cost], record["ceiling_db"]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                fontsize=6.5,
                color=INK,
            )
    front = pareto(records, cost)
    xs = [r[cost] for r in front]
    ys = [r["ceiling_db"] for r in front]
    ax.step(xs, ys, where="post", color=INK, lw=0.6, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ceiling (dB, noiseless input)")
    ax.set_ylim(0, 6.6)


def figure(mit_records: list[Record]) -> None:
    mpl.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * SINGLE_COLUMN_IN, 2.3), sharey=True, layout="constrained"
    )
    draw(axes[0], mit_records, "synops_per_s", "Synaptic operations per second")
    draw(axes[1], mit_records, "synapses", "Synapses")
    axes[1].set_ylabel("")
    axes[0].set_title("(a) Event-driven cost", loc="left")
    axes[1].set_title("(b) Clock-driven cost (memory)", loc="left")
    present = {r["lever"] for r in mit_records}
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=marker,
            markersize=7 if marker == "*" else 4,
            color=colour,
            markerfacecolor=colour,
            markeredgecolor="white" if marker != "x" else colour,
            label=lever,
        )
        for lever, (colour, marker) in LEVER_STYLE.items()
        if lever in present
    ]
    handles.append(Line2D([], [], color=INK, lw=0.6, ls="--", label="Pareto front"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "pilot_b_front"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.pdf (+ .png preview)")


def main() -> None:
    r3.OUT = ANALYSIS
    print("H1 pilot B — cost of every measured design point")
    points = collect()
    records = summarise(points)
    r3.write("pilot_b_points.csv", records)

    mit_records = [r for r in records if r["signal"] == "mit-bih"]
    fronts: list[Record] = []
    for cost in ("synops_per_s", "synapses", "spikes_per_s"):
        for r in pareto(mit_records, cost):
            fronts.append({"cost": cost, **r})
        print(f"\n  MIT-BIH front on {cost}:")
        for r in pareto(mit_records, cost):
            print(
                f"    {r['design']:<34} {r['lever']:<17} "
                f"{r['ceiling_db']:5.2f} dB  {r[cost]:>10.4g}"
            )
    r3.write("pilot_b_fronts.csv", fronts)

    print("\n  Paired contrasts (MIT-BIH, other minus reference):")
    paired = contrasts(points)
    for c in paired:
        print(
            f"    {c['contrast']:<48} {r3.fmt(c['median_diff'])} dB "
            f"[{r3.fmt(c['ci_low'])}, {r3.fmt(c['ci_high'])}] p {c['p_value']:.2g}; "
            f"synops x{c['synops_per_s_ratio']:.3g}, "
            f"synapses x{c['synapses_ratio']:.3g}, "
            f"spikes x{c['spikes_per_s_ratio']:.3g}"
        )
    r3.write("pilot_b_contrasts.csv", paired)

    print()
    for line in reading(points, pareto(mit_records, "synops_per_s"), mit_records):
        print("  " + line)
    figure(mit_records)


if __name__ == "__main__":
    main()
