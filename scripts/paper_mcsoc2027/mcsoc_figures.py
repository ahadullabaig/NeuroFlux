"""
Figures and tables for the MCSoC 2027 paper (`paper/mcsoc2027/`).

Every panel and every table cell is read from a committed H1 analysis in
`results/h1_hardware_cost/analysis/`, so every number in the paper is a number
in `docs/h1_findings.md`:

  fig_landscape    Fig. 2, cost against fidelity       pilot B, B2 (§1, §3.2)
  fig_pool         Fig. 3, pool, encoder, law check    B2 (§3.1, §3.3)
  fig_break_even   Fig. 4, event cost against rho      confirmation, break-even (§7, §8)
  fig_noisy        Fig. 5, named designs, noisy input  confirmation grid (§7)
  fig_devices      Fig. 6, precision and device faults pilot A (§6)
  table_levers     Table I, what each lever costs      pilot B, B2
  table_designs    Table II, the named designs         confirmation grid (§7)
  table_devices    Table III, precision and fault budget  pilot A

The paper's Fig. 1 (the pipeline and its costs) is TikZ inside `main.tex`.

Style is R3's (`figures_r3.py`): vector PDF with TrueType fonts for IEEE PDF
eXpress, Okabe-Ito hues, and markers as well as colours so the figures survive
greyscale.

Usage:
    uv run python scripts/paper_mcsoc2027/mcsoc_figures.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from figures_r3 import (  # noqa: E402
    BLUE,
    DOUBLE_COLUMN_IN,
    GREEN,
    IEEE_RC,
    INK,
    REFERENCE,
    SINGLE_COLUMN_IN,
    VERMILLION,
    WINE,
)

PAPER = Path("paper/mcsoc2027")
FIGURES = PAPER / "figures"
ANALYSIS = Path("results/h1_hardware_cost/analysis")

Row = dict[str, str]

# The three designs the paper names on the landscape, as labelled there.
NAMED = {
    "[32,16,8] @40;40": "shipped architecture",
    "[512,512] K=16 @20": "previous best",
    "[512,128] K=8 enc 40 Hz": "this work",
}
POOL_COLOURS = {128: VERMILLION, 512: BLUE, 2048: INK}


def load(name: str) -> list[Row]:
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str) -> float:
    return float(value) if value else math.nan


def save(fig: Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {FIGURES / stem}.pdf")


def signed(value: float, digits: int = 2) -> str:
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    text = f"{value:+.{digits}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def times(ratio: float) -> str:
    if ratio >= 10:
        return f"{ratio:.1f}"
    return f"{ratio:.2f}" if ratio >= 0.1 else f"{ratio:.3f}"


def interval(row: Row) -> str:
    return (
        f"{signed(fnum(row['median_diff']))} "
        f"[{signed(fnum(row['ci_low']))}, {signed(fnum(row['ci_high']))}]"
    )


# --------------------------------------------------------------------------
# Fig. 2 — the cost-fidelity landscape
# --------------------------------------------------------------------------


def landscape_points() -> tuple[list[dict[str, Any]], list[Row]]:
    points: list[dict[str, Any]] = []
    for r in load("pilot_b_points.csv"):
        if r["signal"] == "mit-bih":
            points.append(
                {
                    "design": r["design"],
                    "source": "earlier",
                    "one_layer": r["layers"].count("-") == 1,
                    "ceiling": fnum(r["ceiling_db"]),
                    "synops": fnum(r["synops_per_s"]),
                    "spikes": fnum(r["spikes_per_s"]),
                }
            )
    for r in load("b2_cells.csv"):
        points.append(
            {
                "design": r["design"],
                "source": "B2",
                "one_layer": True,
                "ceiling": fnum(r["ceiling_db"]),
                "synops": fnum(r["synops_per_s"]),
                "spikes": fnum(r["spikes_per_s"]),
            }
        )
    return points, load("b2_front.csv")


def landscape_panel(
    ax: Axes, points: list[dict[str, Any]], front: list[Row], cost: str
) -> None:
    key = "synops" if cost == "synops_per_s" else "spikes"
    for p in points:
        if p["source"] == "B2":
            marker, size, colour = "D", 9.0, BLUE
        elif p["one_layer"]:
            marker, size, colour = "s", 8.0, GREEN
        else:
            marker, size, colour = "o", 7.0, REFERENCE
        ax.scatter(
            p[key],
            p["ceiling"],
            marker=marker,
            s=size,
            color=colour,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    rows = sorted((r for r in front if r["cost"] == cost), key=lambda r: fnum(r[cost]))
    ax.step(
        [fnum(r[cost]) for r in rows],
        [fnum(r["ceiling_db"]) for r in rows],
        where="post",
        color=INK,
        lw=0.6,
        ls="--",
        zorder=2,
    )
    offsets = (
        {
            "shipped architecture": (4, -8),
            "previous best": (-4, 5),
            "this work": (-4, 5),
        }
        if key == "synops"
        else {
            "shipped architecture": (4, -8),
            "previous best": (-5, 6),
            "this work": (-5, -10),
        }
    )
    for p in points:
        name = NAMED.get(str(p["design"]))
        if name is None:
            continue
        ax.scatter(
            p[key],
            p["ceiling"],
            s=26,
            facecolors="none",
            edgecolors=WINE,
            linewidths=0.8,
            zorder=4,
        )
        dx, dy = offsets[name]
        ax.annotate(
            name,
            (p[key], p["ceiling"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            fontsize=6.5,
            color=WINE,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 6.6)


def figure_landscape() -> None:
    points, front = landscape_points()
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_IN, 2.0), sharey=True, layout="constrained"
    )
    landscape_panel(axes[0], points, front, "synops_per_s")
    landscape_panel(axes[1], points, front, "spikes_per_s")
    axes[0].set_xlabel("Synaptic operations per second")
    axes[1].set_xlabel("Spikes per second (encoder and neurons)")
    axes[0].set_ylabel("Ceiling (dB)")
    axes[0].set_title("(a) Event-driven work", loc="left")
    axes[1].set_title("(b) Address-event traffic", loc="left")
    handles = [
        Line2D([], [], ls="", marker="o", color=REFERENCE, label="two or three layers"),
        Line2D([], [], ls="", marker="s", color=GREEN, label="one layer (R3, R4b)"),
        Line2D(
            [], [], ls="", marker="D", color=BLUE, label="one layer, K = 8 (pool grid)"
        ),
        Line2D([], [], color=INK, ls="--", lw=0.6, label="Pareto front"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=6)
    save(fig, "fig_landscape")


# --------------------------------------------------------------------------
# Fig. 3 — input pool and encoder rate
# --------------------------------------------------------------------------


def figure_pool() -> None:
    cells = load("b2_cells.csv")
    shares = [r for r in load("b2_post_hoc.csv") if r["analysis"].startswith("encoder")]
    law = load("b2_p2b_cells.csv")
    p2b = next(r for r in load("b2_tests.csv") if r["test"].startswith("P2b"))
    fig = plt.figure(figsize=(SINGLE_COLUMN_IN, 3.1), layout="constrained")
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    axes = [
        fig.add_subplot(grid_spec[0, :]),
        fig.add_subplot(grid_spec[1, 0]),
        fig.add_subplot(grid_spec[1, 1]),
    ]
    ax = axes[0]
    styles = {40.0: ("-", "o"), 20.0: ("--", "s"), 10.0: (":", "^")}
    colours = {128: BLUE, 32: VERMILLION}
    for n_1 in (128, 32):
        for hz, (ls, marker) in styles.items():
            rows = sorted(
                (
                    r
                    for r in cells
                    if int(r["n_1"]) == n_1 and fnum(r["encoder_hz"]) == hz
                ),
                key=lambda r: int(r["n_in"]),
            )
            ax.plot(
                [int(r["n_in"]) for r in rows],
                [fnum(r["ceiling_db"]) for r in rows],
                color=colours[n_1],
                ls=ls,
                marker=marker,
                ms=2.4,
                lw=0.8,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
    ax.set_xlabel("Input pool $N_\\mathrm{in}$")
    ax.set_ylabel("Ceiling (dB)")
    ax.set_title("(a)", loc="left")
    handles = [
        Line2D([], [], color=colours[n], label=f"$N_1$ = {n}") for n in (128, 32)
    ] + [
        Line2D([], [], color=INK, ls=ls, marker=m, ms=2.4, lw=0.8, label=f"{hz:g} Hz")
        for hz, (ls, m) in styles.items()
    ]
    ax.set_ylim(2.0, 6.2)
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=5.5,
        ncol=5,
        handlelength=1.8,
        columnspacing=0.8,
    )

    ax = axes[1]
    for n_1, marker in ((128, "o"), (32, "s")):
        rows = [r for r in shares if r["design"].endswith(f",{n_1}] K=8")]
        rows.sort(key=lambda r: int(r["design"][1:].split(",")[0]))
        ax.plot(
            [int(r["design"][1:].split(",")[0]) for r in rows],
            [100 * fnum(r["encoder_share_median"]) for r in rows],
            color=colours[n_1],
            marker=marker,
            ms=2.6,
            lw=0.8,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
    ax.set_ylim(0, 50)
    ax.set_xlabel("Input pool $N_\\mathrm{in}$")
    ax.set_ylabel("Encoder share of noise (%)")
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    for n_1, marker in ((128, "o"), (32, "s")):
        rows = [r for r in law if int(r["n_1"]) == n_1]
        ax.scatter(
            [fnum(r["predicted_10hz_db"]) for r in rows],
            [fnum(r["measured_10hz_db"]) for r in rows],
            s=5,
            marker=marker,
            color=colours[n_1],
            edgecolors="none",
            alpha=0.8,
            zorder=3,
        )
    values = [
        fnum(r[k]) for r in law for k in ("predicted_10hz_db", "measured_10hz_db")
    ]
    low, high = min(values) - 0.2, max(values) + 0.2
    ax.plot([low, high], [low, high], color=REFERENCE, lw=0.6, ls="--", zorder=2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Predicted at 10 Hz (dB)")
    ax.set_ylabel("Measured at 10 Hz (dB)")
    ax.set_title("(c)", loc="left")
    ax.text(
        0.96,
        0.05,
        f"median |error|\n{fnum(p2b['median_abs_error_db']):.3f} dB",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    save(fig, "fig_pool")


# --------------------------------------------------------------------------
# Fig. 4 — precision and device faults
# --------------------------------------------------------------------------


def figure_devices() -> None:
    cells = load("pilot_a_cells.csv")

    def change(n_in: int, hz: float, level: str) -> float:
        row = next(
            r
            for r in cells
            if int(r["n_in"]) == n_in
            and fnum(r["encoder_hz"]) == hz
            and r["level"] == level
        )
        return fnum(row["change_db"])

    axes_spec = [
        (
            "(a) Weight bits",
            [
                ("exact", "32"),
                ("4 bit", "4"),
                ("3 bit", "3"),
                ("2 bit", "2"),
                ("1 bit", "1"),
            ],
        ),
        (
            "(b) Variation $\\sigma$",
            [
                ("exact", "0"),
                ("sigma 0.25", ".25"),
                ("sigma 0.5", ".5"),
                ("sigma 1", "1"),
                ("sigma 1.5", "1.5"),
            ],
        ),
        (
            "(c) Stuck at zero",
            [
                ("exact", "0"),
                ("stuck 0.25", "25%"),
                ("stuck 0.5", "50%"),
                ("stuck 0.75", "75%"),
            ],
        ),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(SINGLE_COLUMN_IN, 1.7), sharey=True, layout="constrained"
    )
    for ax, (title, levels) in zip(axes, axes_spec, strict=True):
        xs = list(range(len(levels)))
        ax.axhspan(-0.5, 0.5, color="#eeeeee", zorder=0, lw=0)
        for n_in, colour in POOL_COLOURS.items():
            for hz, ls in ((40.0, "-"), (10.0, ":")):
                ax.plot(
                    xs,
                    [change(n_in, hz, lv) for lv, _ in levels],
                    color=colour,
                    ls=ls,
                    marker="o",
                    ms=2.0,
                    lw=0.8,
                )
        ax.set_xticks(xs, [label for _, label in levels])
        ax.tick_params(axis="x", labelsize=6)
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_ylim(-0.8, 0.15)
    axes[0].set_ylabel("Change from exact (dB)")
    handles = [
        Line2D([], [], color=c, lw=0.8, label=f"$N_\\mathrm{{in}}$ = {n}")
        for n, c in POOL_COLOURS.items()
    ] + [
        Line2D([], [], color=INK, ls="-", lw=0.8, label="40 Hz"),
        Line2D([], [], color=INK, ls=":", lw=0.8, label="10 Hz"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        fontsize=6,
        handlelength=1.6,
        columnspacing=0.8,
    )
    save(fig, "fig_devices")


# --------------------------------------------------------------------------
# Fig. 5 — event cost against the spike-to-synaptic-operation energy ratio
# --------------------------------------------------------------------------


def figure_break_even() -> None:
    designs = {
        r["design"]: r for r in load("break_even_designs.csv") if r.get("ceiling_db")
    }
    labels = {name: (label, c, ls) for name, (label, c, ls, _) in DESIGN_STYLE.items()}
    rhos = [10 ** (k / 20) for k in range(-40, 41)]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 1.9), layout="constrained")
    for name, (label, colour, ls) in labels.items():
        d = designs[name]
        synops, spikes = fnum(d["synops_per_s"]), fnum(d["spikes_per_s"])
        ax.plot(
            rhos,
            [synops + rho * spikes for rho in rhos],
            color=colour,
            ls=ls,
            lw=0.9,
            label=f"{label}, {fnum(d['ceiling_db']):.2f} dB",
        )
    crossing = next(
        r
        for r in load("break_even_designs.csv")
        if r.get("design_a") == "D4" and r.get("design_b") == "D5"
    )
    rho_star = fnum(crossing["crossover_rho"])
    d5 = designs["D5"]
    ax.scatter(
        [rho_star],
        [fnum(d5["synops_per_s"]) + rho_star * fnum(d5["spikes_per_s"])],
        s=14,
        color=INK,
        zorder=4,
    )
    ax.annotate(
        f"$\\rho^* = {rho_star:.2f}$",
        (rho_star, fnum(d5["synops_per_s"]) + rho_star * fnum(d5["spikes_per_s"])),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\rho = E_\\mathrm{spike} / E_\\mathrm{synop}$")
    ax.set_ylabel("Synaptic-op equivalents / s")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8)
    save(fig, "fig_break_even")


# --------------------------------------------------------------------------
# Fig. 6 — the named designs at noisy inputs
# --------------------------------------------------------------------------

DESIGN_STYLE: dict[str, tuple[str, str, str, str]] = {
    "D1": ("previous best", REFERENCE, "--", "s"),
    "D2": ("this work", WINE, "-", "o"),
    "D4": ("wide pool, 10 Hz", BLUE, "-", "^"),
    "D5": ("narrow pool, 40 Hz", VERMILLION, "-", "v"),
    "D0": ("shipped arch.", INK, ":", "x"),
}


def figure_noisy() -> None:
    cells = {
        (r["signal"], r["input_snr"], r["design"]): fnum(r["ceiling_db"])
        for r in load("confirm_cells.csv")
    }
    inputs = [("0.0", "0"), ("10.0", "10"), ("inf", "$\\infty$")]
    signals = [("mit-bih", "(a) MIT-BIH"), ("eeg", "(b) EEG"), ("sine", "(c) Sine")]
    fig, axes = plt.subplots(
        1, 3, figsize=(SINGLE_COLUMN_IN, 1.75), layout="constrained"
    )
    for ax, (signal, title) in zip(axes, signals, strict=True):
        xs = list(range(len(inputs)))
        for name, (_, colour, ls, marker) in DESIGN_STYLE.items():
            ax.plot(
                xs,
                [cells[(signal, snr, name)] for snr, _ in inputs],
                color=colour,
                ls=ls,
                marker=marker,
                ms=2.4,
                lw=0.8,
            )
        ax.set_xticks(xs, [label for _, label in inputs])
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlabel("Input SNR (dB)")
    axes[0].set_ylabel("Output SNR (dB)")
    handles = [
        Line2D([], [], color=c, ls=ls, marker=m, ms=2.4, lw=0.8, label=label)
        for label, c, ls, m in DESIGN_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        fontsize=5.8,
        handlelength=1.8,
        columnspacing=0.8,
    )
    save(fig, "fig_noisy")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def write_table(stem: str, text: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / f"{stem}.tex").write_text(text, encoding="utf-8")
    print(f"  wrote {FIGURES / stem}.tex")


def table_levers() -> None:
    contrasts = {r["contrast"]: r for r in load("pilot_b_contrasts.csv")}
    b2 = {r["test"]: r for r in load("b2_tests.csv")}
    cells = {
        (int(r["n_in"]), int(r["n_1"]), fnum(r["encoder_hz"])): r
        for r in load("b2_cells.csv")
    }

    def ratio(a: tuple[int, int, float], b: tuple[int, int, float], key: str) -> float:
        return fnum(cells[b][key]) / fnum(cells[a][key])

    rows: list[tuple[str, str, str, str]] = []
    for label, name in (
        ("Second LIF layer, $[32,8]\\to[32,16,8]$", "depth 2 vs 1 LIF layer"),
        ("Third LIF layer", "depth 3 vs 2 LIF layers"),
        ("10 STDP epochs, rates held", "10 STDP epochs vs none, held rate"),
        ("Output rate 40 $\\to$ 5 Hz", "output rate 5 Hz vs 40 Hz"),
        ("Output width 8 $\\to$ 64", "output width 64 vs 8"),
        ("Input width 32 $\\to$ 256, dense", "input width 256 vs 32, output 40 Hz"),
        ("Fan-in dense $\\to$ 16, $[512,512]$", "[512,512] K=16 vs dense, 20 Hz"),
    ):
        r = contrasts[name]
        rows.append(
            (
                label,
                interval(r),
                times(fnum(r["synops_per_s_ratio"])),
                times(fnum(r["spikes_per_s_ratio"])),
            )
        )
    f1 = b2["F1 [2048,128] minus [128,128], 40 Hz"]
    rows.append(
        (
            "Input pool 128 $\\to$ 2048, $K=8$",
            interval(f1),
            times(ratio((128, 128, 40.0), (2048, 128, 40.0), "synops_per_s")),
            times(ratio((128, 128, 40.0), (2048, 128, 40.0), "spikes_per_s")),
        )
    )
    p2a = b2["P2a 40 Hz minus 10 Hz encoder, [128,128]"]
    reversed_row = {
        "median_diff": str(-fnum(p2a["median_diff"])),
        "ci_low": str(-fnum(p2a["ci_high"])),
        "ci_high": str(-fnum(p2a["ci_low"])),
    }
    rows.append(
        (
            "Encoder 40 $\\to$ 10 Hz, $[128,128]$",
            interval(reversed_row),
            times(ratio((128, 128, 40.0), (128, 128, 10.0), "synops_per_s")),
            times(ratio((128, 128, 40.0), (128, 128, 10.0), "spikes_per_s")),
        )
    )
    body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
    write_table(
        "table_levers",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{What each lever buys and costs on MIT-BIH}\n"
        "\\label{tab:levers}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Change & $\\Delta$ceiling, dB [95\\% CI] & Synops & Spikes \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{2pt}\n\\parbox{\\columnwidth}{\\scriptsize Paired by record "
        "($n=16$), noiseless input; cost columns are ratios after/before. "
        "Rows 1--7 from the earlier grids, rows 8--9 from the pool grid.}\n"
        "\\end{table}\n",
    )


def table_designs() -> None:
    cells = {
        (r["signal"], r["input_snr"], r["design"]): r for r in load("confirm_cells.csv")
    }
    spec = {
        "D0": ("$[32,16,8]$", "all", "40", "--"),
        "D1": ("$[512,512]$", "16", "40", "--"),
        "D2": ("$[512,128]$", "8", "40", "--"),
        "D3": ("$[512,128]$", "8", "40", "2"),
        "D4": ("$[2048,128]$", "8", "10", "--"),
        "D5": ("$[128,128]$", "8", "40", "--"),
    }
    rows = []
    for name, (net, k, enc, bits) in spec.items():
        m = cells[("mit-bih", "inf", name)]
        rows.append(
            f"{name} & {net} & {k} & {enc} & {bits} & "
            f"{fnum(m['ceiling_db']):.2f} & "
            f"{fnum(cells[('eeg', 'inf', name)]['ceiling_db']):.2f} & "
            f"{fnum(cells[('sine', 'inf', name)]['ceiling_db']):.2f} & "
            f"{fnum(m['synops_per_s']) / 1000:.1f} & "
            f"{fnum(m['spikes_per_s']) / 1000:.1f} & "
            f"{fnum(m['synapses']):,.0f} & "
            f"{fnum(m['weight_memory_bits']) / 1000:.1f} & "
            f"{fnum(m['clock_synapse_reads_per_s']) / 1000:,.0f} \\\\"
        )
    tests = {r["test"]: r for r in load("confirm_tests.csv")}
    c1 = tests["C1 headline, D2 vs D1"]
    c2 = tests["C2 two bits, D3 vs D2"]
    c3 = tests["C3 break-even pair, D4 vs D5 (non-inferiority)"]
    note = (
        f"D2 minus D1 {interval(c1)}~dB; D3 minus D2 {interval(c2)}; "
        f"D4 minus D5 {interval(c3)}. Medians over 16 records (MIT-BIH) or 10 "
        "seeds, seeds 10--19, noiseless input. Synops and spikes: thousands per "
        "second, measured. Memory: kilobits of weights, at 32 bits unless "
        "quantized, plus an input address per synapse of a fixed-fan-in layer. "
        "Clock: thousands of synapse reads per second at 250 steps per second."
    )
    write_table(
        "table_designs",
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{The named designs, re-measured on unseen data}\n"
        "\\label{tab:designs}\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\begin{tabular}{@{}llcccccc rrrrr@{}}\n\\toprule\n"
        " & & & Enc. & Weight & \\multicolumn{3}{c}{Ceiling (dB)} & "
        "\\multicolumn{2}{c}{Event-driven} & \\multicolumn{3}{c}{Static} \\\\\n"
        "\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(l){11-13}\n"
        " & Network & Fan-in & (Hz) & bits & MIT-BIH & EEG & Sine & "
        "Synops & Spikes & Synapses & Memory & Clock \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{2pt}\n{\\scriptsize " + note + "}\n\\end{table*}\n",
    )


def table_devices() -> None:
    cells = load("pilot_a_cells.csv")
    order = [
        ("1 bit", "1 bit"),
        ("2 bit", "2 bits"),
        ("3 bit", "3 bits"),
        ("4 bit", "4 bits"),
        ("sigma 0.25", "$\\sigma = 0.25$"),
        ("sigma 0.5", "$\\sigma = 0.5$"),
        ("sigma 1", "$\\sigma = 1$"),
        ("sigma 1.5", "$\\sigma = 1.5$"),
        ("stuck 0.25", "25\\% stuck"),
        ("stuck 0.5", "50\\% stuck"),
        ("stuck 0.75", "75\\% stuck"),
    ]
    rows = []
    for level, label in order:
        group = [r for r in cells if r["level"] == level]
        measured = [fnum(r["change_db"]) for r in group]
        predicted = [fnum(r["predicted_change_db"]) for r in group]
        equivalent = sum(r["equivalent_to_exact"] == "True" for r in group)
        rows.append(
            f"{label} & {signed(min(measured))} to {signed(max(measured))} & "
            f"{signed(min(predicted))} to {signed(max(predicted))} & "
            f"{equivalent}/{len(group)} \\\\"
        )
    write_table(
        "table_devices",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Precision and fault budget}\n\\label{tab:devices}\n"
        "\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Device level & Measured (dB) & Predicted (dB) & Equivalent \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{2pt}\n\\parbox{\\columnwidth}{\\scriptsize Change from exact "
        "weights, range over three pools and three encoder rates (9 points), "
        "each a median over 16 records; equivalent: TOST $\\pm0.5$~dB. "
        "Predicted: effective fan-in model, fixed in advance.}\n\\end{table}\n",
    )


def main() -> None:
    mpl.rcParams.update(IEEE_RC)
    print("MCSoC 2027 figures and tables")
    figure_landscape()
    figure_pool()
    figure_devices()
    figure_break_even()
    figure_noisy()
    table_levers()
    table_designs()
    table_devices()


if __name__ == "__main__":
    main()
