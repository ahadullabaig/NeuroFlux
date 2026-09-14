"""
R3 figures and Table I, in IEEE Transactions format.

Every figure is drawn from the CSVs `analyse_r3.py` writes, never from raw
runs, so a number in a figure is the same number as in the findings document.

Format: vector PDF with TrueType-embedded fonts (IEEE PDF eXpress rejects
Type 3), Times-metric serif text at 8 pt, 3.5 in single-column and 7.16 in
double-column widths. Series are told apart by marker as well as colour, so
they survive greyscale printing; the four colours are Okabe-Ito hues checked
with a colour-vision-deficiency validator (worst adjacent deutan ΔE 10.3).

Usage:
    uv run python scripts/r3_characterisation/figures_r3.py
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patches  # noqa: E402, F401
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ANALYSIS = Path("results/r3_characterisation/analysis")
FIGURES = Path("results/r3_characterisation/figures")

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16

IEEE_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Liberation Serif",
        "Nimbus Roman",
        "STIX Two Text",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.color": "#dcdcdc",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Okabe-Ito blue, vermillion, bluish green, and a wine for the fourth slot,
# which clears the deutan separation floor where reddish purple did not.
BLUE, VERMILLION, GREEN, WINE = "#0072B2", "#D55E00", "#009E73", "#882255"
INK, REFERENCE = "#222222", "#8c8c8c"

ARM_STYLE: dict[str, tuple[str, str, str]] = {
    "controlled-passive": (BLUE, "o", "Rate-controlled, untrained"),
    "shipped-passive": (VERMILLION, "s", "Shipped, untrained"),
    "shipped-trained": (GREEN, "^", "Shipped, 10 STDP epochs"),
}
SIGNAL_TITLE: dict[str, str] = {
    "synthetic-ecg": "(a) Synthetic ECG",
    "mit-bih": "(b) MIT-BIH ECG, 16 records",
    "eeg": "(c) Synthetic EEG",
    "sine": "(d) 2 Hz sine",
}


def load(name: str) -> list[dict[str, str]]:
    path = ANALYSIS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str) -> float:
    return float(value) if value else math.nan


def save(fig: Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {FIGURES / stem}.pdf (+ .png preview)")


def sorted_by(rows: Iterable[dict[str, str]], key: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: fnum(r[key]))


# --------------------------------------------------------------------------
# Fig. 1 — output SNR against input SNR
# --------------------------------------------------------------------------

# Where the noiseless input is drawn, past the end of the finite axis.
CLEAN_X = 47.5


def figure_1() -> None:
    curve = load("c1_curve.csv")
    if not curve:
        print("  Fig. 1 skipped: run analyse_r3.py --part c1 first")
        return

    fig, axes = plt.subplots(
        1, 4, figsize=(DOUBLE_COLUMN_IN, 2.05), constrained_layout=True
    )
    for ax, signal in zip(axes, SIGNAL_TITLE, strict=True):
        rows = [r for r in curve if r["signal"] == signal]
        ceiling_top = max(
            fnum(r["output_affine_q75"])
            for r in rows
            if math.isinf(fnum(r["input_snr_db"]))
        )
        y_top = ceiling_top * 1.35 + 0.4

        reference = sorted_by(
            (
                r
                for r in rows
                if r["arm"] == "controlled-passive"
                and math.isfinite(fnum(r["input_snr_db"]))
            ),
            "input_snr_db",
        )
        ax.plot(
            [fnum(r["input_snr_db"]) for r in reference],
            [fnum(r["input_affine_median"]) for r in reference],
            color=REFERENCE,
            linewidth=0.8,
            zorder=1,
        )

        for offset, (arm, (color, marker, _)) in zip(
            (-1.6, 0.0, 1.6), ARM_STYLE.items(), strict=True
        ):
            block = sorted_by((r for r in rows if r["arm"] == arm), "input_snr_db")
            finite = [r for r in block if math.isfinite(fnum(r["input_snr_db"]))]
            clean = [r for r in block if math.isinf(fnum(r["input_snr_db"]))]
            x = [fnum(r["input_snr_db"]) for r in finite]
            ax.fill_between(
                x,
                [fnum(r["output_affine_q25"]) for r in finite],
                [fnum(r["output_affine_q75"]) for r in finite],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                x,
                [fnum(r["output_affine_median"]) for r in finite],
                color=color,
                marker=marker,
                markevery=2,
                zorder=3,
            )
            if clean:
                c = clean[0]
                ax.errorbar(
                    [CLEAN_X + offset],
                    [fnum(c["output_affine_median"])],
                    yerr=[
                        [
                            fnum(c["output_affine_median"])
                            - fnum(c["output_affine_q25"])
                        ],
                        [
                            fnum(c["output_affine_q75"])
                            - fnum(c["output_affine_median"])
                        ],
                    ],
                    color=color,
                    marker=marker,
                    capsize=1.5,
                    elinewidth=0.6,
                    zorder=3,
                )
            if arm == "controlled-passive":
                ax.plot(
                    x,
                    [fnum(r["predicted_median"]) for r in finite],
                    color=INK,
                    linestyle=(0, (2.2, 1.6)),
                    linewidth=0.7,
                    zorder=4,
                )

        ax.axvline(42.5, color="#bdbdbd", linewidth=0.4, zorder=0)
        ax.set_xlim(-22, 50)
        ax.set_ylim(0, y_top)
        ax.set_xticks([-20, 0, 20, 40, CLEAN_X])
        ax.set_xticklabels(["−20", "0", "20", "40", "∞"])
        ax.set_title(SIGNAL_TITLE[signal], loc="left", pad=3)
        ax.set_xlabel("Input SNR (dB)")
    axes[0].set_ylabel("Output SNR, affine-corrected (dB)")

    handles = [
        Line2D([], [], color=REFERENCE, linewidth=0.8, label="Unprocessed input"),
        *[
            Line2D([], [], color=c, marker=m, label=label)
            for c, m, label in ARM_STYLE.values()
        ],
        Line2D(
            [],
            [],
            color=INK,
            linestyle=(0, (2.2, 1.6)),
            linewidth=0.7,
            label="Cascade law, rate-controlled",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=5,
        handlelength=2.2,
        columnspacing=1.2,
    )
    save(fig, "fig1_flatness")


# --------------------------------------------------------------------------
# Fig. 2 and Table I — invariances
# --------------------------------------------------------------------------

KNOB_AXIS: dict[str, tuple[str, str, bool]] = {
    # knob: (panel title, x label, log x)
    "output_width": ("(a) Output width", "Output neurons $N$", True),
    "decoder_tau": ("(b) Decoder time constant", "$\\tau_\\mathrm{dec}$ (ms)", True),
    "epochs": ("(c) Training", "STDP epochs", False),
    "depth": ("(d) Depth", "Layers", False),
    "stdp_tau": ("(e) STDP time constant", "$\\tau_\\mathrm{STDP}$ (ms)", True),
}
# (signal, operating point, colour, marker, label)
C2_SERIES: tuple[tuple[str, str, str, str, str], ...] = (
    ("mit-bih", "controlled", BLUE, "o", "MIT-BIH, rate-controlled"),
    ("synthetic-ecg", "controlled", VERMILLION, "s", "Synthetic ECG, rate-controlled"),
    ("mit-bih", "shipped", GREEN, "^", "MIT-BIH, shipped (rates free)"),
)


def epoch_position(level: float) -> float:
    """Epochs 0, 10, 50, 200 at evenly spaced positions."""
    return {0.0: 0.0, 10.0: 1.0, 50.0: 2.0, 200.0: 3.0}.get(level, math.nan)


def figure_2() -> None:
    comparisons = load("c2_comparisons.csv")
    widths = load("c2_width_prediction.csv")
    if not comparisons:
        print("  Fig. 2 skipped: run analyse_r3.py --part c2 first")
        return

    # One shared y-axis, so the equivalence band and every effect are drawn to
    # the same scale and the knobs can be compared by eye.
    fig, axes = plt.subplots(
        1, 5, figsize=(DOUBLE_COLUMN_IN, 1.95), constrained_layout=True, sharey=True
    )
    for ax, (knob, (title, xlabel, log_x)) in zip(axes, KNOB_AXIS.items(), strict=True):
        ax.axhspan(-0.5, 0.5, color="#ececec", linewidth=0, zorder=0)
        ax.axhline(0.0, color=REFERENCE, linewidth=0.5, zorder=1)
        rows = [
            r
            for r in comparisons
            if r["knob"] == knob
            and r["input"] == "clean"
            and r["metric"] == "snr_affine_db"
        ]
        spread = (-0.12, 0.0, 0.12) if not log_x else (0.93, 1.0, 1.075)
        for index, (signal, operating_point, color, marker, _) in enumerate(C2_SERIES):
            block = [
                r
                for r in rows
                if r["signal"] == signal and r["operating_point"] == operating_point
            ]
            if not block:
                continue
            reference = fnum(block[0]["reference_level"])
            points = sorted(
                [(reference, 0.0, 0.0, 0.0)]
                + [
                    (
                        fnum(r["level"]),
                        fnum(r["median_diff"]),
                        fnum(r["ci_low"]),
                        fnum(r["ci_high"]),
                    )
                    for r in block
                ]
            )
            if knob == "epochs":
                xs = [epoch_position(p[0]) + spread[index] for p in points]
            elif log_x:
                xs = [p[0] * spread[index] for p in points]
            else:
                xs = [p[0] + spread[index] for p in points]
            ys = [p[1] for p in points]
            low = [p[1] - p[2] if math.isfinite(p[2]) else 0.0 for p in points]
            high = [p[3] - p[1] if math.isfinite(p[3]) else 0.0 for p in points]
            ax.errorbar(
                xs,
                ys,
                yerr=[low, high],
                color=color,
                marker=marker,
                linewidth=0.8,
                elinewidth=0.6,
                capsize=1.2,
                zorder=3,
            )

        if knob == "output_width" and widths:
            predicted: list[tuple[float, float]] = []
            units = {
                r["unit"]
                for r in widths
                if r["signal"] == "mit-bih" and r["unit"] != "summary"
            }
            for width in sorted({fnum(r["width"]) for r in widths if r.get("width")}):
                deltas = []
                for unit in units:
                    ref = [
                        r
                        for r in widths
                        if r["signal"] == "mit-bih"
                        and r["unit"] == unit
                        and fnum(r["width"]) == 8.0
                    ]
                    at = [
                        r
                        for r in widths
                        if r["signal"] == "mit-bih"
                        and r["unit"] == unit
                        and fnum(r["width"]) == width
                    ]
                    if ref and at:
                        deltas.append(
                            fnum(at[0]["capacity_predicted_ceiling"])
                            - fnum(ref[0]["measured_ceiling"])
                        )
                finite = sorted(d for d in deltas if math.isfinite(d))
                if finite:
                    predicted.append((width, finite[len(finite) // 2]))
            if predicted:
                ax.plot(
                    [w for w, _ in predicted],
                    [d for _, d in predicted],
                    color=INK,
                    linestyle=(0, (2.2, 1.6)),
                    linewidth=0.7,
                    zorder=2,
                )

        if log_x:
            ax.set_xscale("log", base=2)
            levels = sorted(
                {fnum(r["level"]) for r in rows}
                | {fnum(r["reference_level"]) for r in rows}
            )
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{v:g}" for v in levels])
            ax.minorticks_off()
        elif knob == "epochs":
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["0", "10", "50", "200"])
        else:
            ax.set_xticks([2, 3, 4])
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Change in ceiling (dB)")
    axes[0].set_ylim(-1.65, 1.3)

    handles = [
        Line2D([], [], color=c, marker=m, label=label)
        for _, _, c, m, label in C2_SERIES
    ] + [
        Line2D(
            [],
            [],
            color=INK,
            linestyle=(0, (2.2, 1.6)),
            linewidth=0.7,
            label="Independent-neuron prediction",
        ),
        mpl.patches.Patch(color="#ececec", label="±0.5 dB equivalence margin"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, columnspacing=1.0)
    save(fig, "fig2_invariances")


def verdict_mark(row: dict[str, str]) -> str:
    """
    Table I's reading of one comparison.

    Equivalence is checked first: an effect shown to lie inside the margin is
    negligible even when it is also significantly non-zero. An arrow marks a
    significant effect that TOST could not confine to the margin, whatever its
    point estimate.
    """
    if row["equivalent"] == "True":
        return "$\\equiv$"
    if row["significant"] == "True":
        return "$\\uparrow$" if fnum(row["median_diff"]) > 0 else "$\\downarrow$"
    return "--"


def table_1() -> None:
    comparisons = load("c2_comparisons.csv")
    if not comparisons:
        print("  Table I skipped: run analyse_r3.py --part c2 first")
        return

    columns = (
        ("mit-bih", "clean", "snr_affine_db", "MIT-BIH ceiling"),
        ("mit-bih", "10 dB", "snr_affine_db", "MIT-BIH, 10 dB in"),
        ("synthetic-ecg", "clean", "snr_affine_db", "Synth.\\ ECG ceiling"),
        ("mit-bih", "10 dB", "qrs_f1", "QRS $F_1$, 10 dB in"),
    )
    knob_names = {
        "output_width": "Output width $N$",
        "decoder_tau": "Decoder $\\tau$ (ms)",
        "epochs": "STDP epochs",
        "depth": "Layers",
        "stdp_tau": "STDP $\\tau$ (ms)",
    }

    def cell(knob: str, op: str, level: str, column: tuple[str, str, str, str]) -> str:
        signal, input_label, metric, _ = column
        match = [
            r
            for r in comparisons
            if r["knob"] == knob
            and r["operating_point"] == op
            and r["signal"] == signal
            and r["input"] == input_label
            and r["metric"] == metric
            and r["level"] == level
        ]
        if not match:
            return ""
        r = match[0]
        digits = 2
        return (
            f"${fnum(r['median_diff']):+.{digits}f}$ "
            f"[${fnum(r['ci_low']):+.{digits}f}$, ${fnum(r['ci_high']):+.{digits}f}$] "
            f"{verdict_mark(r)}"
        )

    lines = [
        "% Generated by scripts/r3_characterisation/figures_r3.py — do not edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Ceiling invariances under per-layer and encoder rate control. "
        "Median paired change from the reference level [95\\% BCa CI]; units are "
        "records (MIT-BIH, $n=16$) or seeds (synthetic, $n=10$). References: "
        "$N=8$, $\\tau=20$~ms, 0 epochs, 3 layers, STDP $\\tau=20$~ms (trained "
        "10 epochs). Shipped rows let firing rates move with training. "
        "$\\equiv$: equivalent by TOST ($\\pm0.5$~dB; $\\pm0.05$ for $F_1$), "
        "possibly also significantly non-zero. "
        "$\\uparrow\\downarrow$: significant (Wilcoxon, Holm within knob) and not "
        "shown equivalent. --: neither.}",
        "\\label{tab:invariances}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lr" + "l" * len(columns) + "}",
        "\\toprule",
        "Knob & Level & " + " & ".join(c[3] for c in columns) + " \\\\",
        "\\midrule",
    ]
    for knob, name in knob_names.items():
        for op in ("controlled", "shipped"):
            levels = sorted(
                {
                    r["level"]
                    for r in comparisons
                    if r["knob"] == knob and r["operating_point"] == op
                },
                key=fnum,
            )
            if not levels:
                continue
            label = name if op == "controlled" else "STDP epochs, shipped"
            for i, level in enumerate(levels):
                cells = [cell(knob, op, level, c) for c in columns]
                lines.append(
                    f"{label if i == 0 else ''} & {fnum(level):g} & "
                    + " & ".join(cells)
                    + " \\\\"
                )
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "table1_invariances.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# Fig. 3 — the operating region
# --------------------------------------------------------------------------

RATE_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("output 32-16-8", BLUE, "o", "Output layer only, $N=8$"),
    ("output 32-16-64", GREEN, "^", "Output layer only, $N=64$"),
    ("all 32-16-8", VERMILLION, "s", "Every layer, $N=8$"),
)


def plane_panel(ax: Axes, cells: list[dict[str, str]], epochs: str, title: str) -> Any:
    block = [r for r in cells if r["epochs"] == epochs]
    widths = sorted({int(fnum(r["n_in"])) for r in block})
    gains = sorted({fnum(r["g_syn"]) for r in block})
    grid = np.full((len(widths), len(gains)), np.nan)
    state = np.zeros((len(widths), len(gains)))
    for r in block:
        i = widths.index(int(fnum(r["n_in"])))
        j = gains.index(fnum(r["g_syn"]))
        if fnum(r["ok_fraction"]) >= 0.5:
            grid[i, j] = fnum(r["snr_affine_median"])
        elif fnum(r["saturated_fraction"]) >= fnum(r["silent_fraction"]):
            state[i, j] = 2.0
        else:
            state[i, j] = 1.0

    log_g = np.log10(gains)
    step = (log_g[-1] - log_g[0]) / (len(gains) - 1)
    edges_x = 10 ** np.concatenate(([log_g[0] - step / 2], log_g + step / 2))
    edges_y = np.arange(len(widths) + 1) - 0.5
    ax.pcolormesh(
        edges_x,
        edges_y,
        np.where(state == 1.0, 1.0, np.nan),
        cmap=mpl.colors.ListedColormap(["#d9d9d9"]),
    )
    ax.pcolormesh(
        edges_x,
        edges_y,
        np.where(state == 2.0, 1.0, np.nan),
        cmap=mpl.colors.ListedColormap(["#f2c7a8"]),
    )
    mesh = ax.pcolormesh(
        edges_x, edges_y, grid, cmap="Blues", vmin=0.0, vmax=PLANE_VMAX, shading="flat"
    )
    ax.axvline(30.0, color=INK, linewidth=0.7, linestyle=(0, (2.2, 1.6)))
    ax.set_xscale("log")
    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([str(w) for w in widths])
    ax.grid(False)
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel("Global synaptic gain $g_\\mathrm{syn}$")
    return mesh


PLANE_VMAX = 2.75


def figure_3() -> None:
    cells = load("c3_plane_cells.csv")
    rates = load("c3_rate_summary.csv")
    if not cells and not rates:
        print("  Fig. 3 skipped: run analyse_r3.py --part c3-plane / c3-rate first")
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COLUMN_IN, 4.3), constrained_layout=True
    )
    if cells:
        plane_panel(axes[0, 0], cells, "0", "(a) Shipped network, untrained")
        mesh = plane_panel(
            axes[0, 1], cells, "10", "(b) Shipped network, 10 STDP epochs"
        )
        axes[0, 0].set_ylabel("Input neurons $N_\\mathrm{in}$")
        bar = fig.colorbar(mesh, ax=axes[0, :], shrink=0.9, pad=0.01)
        bar.set_label("Output SNR at 10 dB in (dB)")
        bar.ax.tick_params(width=0.4)

    if rates:
        mit = [r for r in rates if r["signal"] == "mit-bih"]
        for scope, color, marker, _ in RATE_SERIES:
            for input_value, style in ((math.inf, "-"), (10.0, (0, (2.2, 1.6)))):
                block = sorted_by(
                    (
                        r
                        for r in mit
                        if r["scope"] == scope
                        and fnum(r["input_snr_db"]) == input_value
                    ),
                    "target_rate_hz",
                )
                x = [fnum(r["out_rate_median"]) for r in block]
                axes[1, 0].plot(
                    x,
                    [fnum(r["snr_affine_median"]) for r in block],
                    color=color,
                    marker=marker,
                    linestyle=style,
                )
                if input_value == 10.0:
                    axes[1, 1].plot(
                        x,
                        [fnum(r["qrs_f1_median"]) for r in block],
                        color=color,
                        marker=marker,
                    )
        f1_input = [
            fnum(r["qrs_f1_input_median"])
            for r in mit
            if fnum(r["input_snr_db"]) == 10.0
        ]
        if f1_input:
            axes[1, 1].axhline(
                sorted(f1_input)[len(f1_input) // 2],
                color=REFERENCE,
                linewidth=0.8,
            )
        for ax in axes[1]:
            ax.set_xscale("log")
            ax.axvline(40.0, color="#bdbdbd", linewidth=0.5, zorder=0)
            ax.set_xlabel("Output-layer firing rate (Hz)")
        axes[1, 0].set_ylabel("Output SNR (dB)")
        axes[1, 0].set_title("(c) Reconstruction, MIT-BIH", loc="left", pad=3)
        axes[1, 1].set_ylabel("QRS $F_1$ at 10 dB input")
        axes[1, 1].set_ylim(0, 1.02)
        axes[1, 1].set_title("(d) Beat detection, MIT-BIH", loc="left", pad=3)
        handles = [
            Line2D([], [], color=c, marker=m, label=label)
            for _, c, m, label in RATE_SERIES
        ] + [
            Line2D([], [], color=INK, label="(c) noiseless input"),
            Line2D(
                [], [], color=INK, linestyle=(0, (2.2, 1.6)), label="(c) 10 dB input"
            ),
            Line2D([], [], color=REFERENCE, label="(d) unprocessed input"),
        ]
        # One legend below every panel: in-panel legends covered data, and
        # in-cell region labels collided with the shipped-gain line.
        legend_handles: list[Artist] = [
            mpl.patches.Patch(color="#d9d9d9", label="(a, b) silent"),
            mpl.patches.Patch(color="#f2c7a8", label="(a, b) saturated"),
            Line2D(
                [],
                [],
                color=INK,
                linestyle=(0, (2.2, 1.6)),
                linewidth=0.7,
                label="(a, b) shipped $g_\\mathrm{syn}=30$",
            ),
            *handles,
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncol=5,
            columnspacing=1.2,
        )
    save(fig, "fig3_operating_region")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    plt.rcParams.update(IEEE_RC)
    print("R3 figures")
    figure_1()
    figure_2()
    table_1()
    figure_3()


if __name__ == "__main__":
    main()
