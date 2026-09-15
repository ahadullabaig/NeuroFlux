"""
Figures and Table I for the ICEE 2026 paper (`paper/icee2026/`).

Single-column IEEE conference floats, drawn from the committed analyses of R3,
R4b and R5 so that every number in the paper is a number in a findings
document:

  fig1_ceiling_law  paper Fig. 2, the ceiling and the law  R3 C1 and its operating point
  fig2_fan_in       paper Fig. 3, fan-in and training      R4b sections 1 and 3.2
  table1_levers     Table I, what moves the ceiling        R3 C2 and C3, R4b, R5 T1

The paper's Fig. 1 (the pipeline) is TikZ inside `main.tex`.

Style (fonts, sizes, colours) is R3's (`figures_r3.py`): vector PDF with
TrueType fonts for IEEE PDF eXpress, Okabe-Ito hues checked for colour-vision
deficiency, and markers as well as colours so the figures survive greyscale.

Usage:
    uv run python scripts/paper_icee2026/figures.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
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

PAPER = Path("paper/icee2026")
FIGURES = PAPER / "figures"
R3_ANALYSIS = Path("results/r3_characterisation/analysis")
R4B_ANALYSIS = Path("results/r4b_mechanism/analysis")
R5_ANALYSIS = Path("results/r5_generality/analysis")

# signal: (colour, marker, legend label)
SIGNALS: dict[str, tuple[str, str, str]] = {
    "mit-bih": (BLUE, "o", "MIT-BIH"),
    "synthetic-ecg": (VERMILLION, "s", "Synth. ECG"),
    "eeg": (GREEN, "^", "Synth. EEG"),
    "sine": (WINE, "D", "Sine"),
}
CLEAN_X = 50.0  # where the noiseless input is drawn, past the finite axis
MARGIN_DB = 0.5


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str) -> float:
    return float(value) if value else math.nan


def save(fig: Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {FIGURES / stem}.pdf (+ .png preview)")


# --------------------------------------------------------------------------
# Fig. 1 — the ceiling, and the law that predicts it
# --------------------------------------------------------------------------


def matched_law_medians() -> dict[str, dict[float, float]]:
    """
    Median over units of the S/N-form law's prediction with intrinsics taken at
    each input's own operating point (R3 section 1.5), rate-controlled arm,
    Gaussian noise. The same computation as `analyse_r3.analyse_c1_operating_point`.
    """
    matched_rows = r3.load(r3.raw("c1_operating_point.csv"))
    c1_rows = r3.load(r3.raw("c1_flatness.csv"))
    medians: dict[str, dict[float, float]] = {}
    for signal in SIGNALS:
        noisy = r3.select(
            c1_rows,
            signal=signal,
            operating_point="controlled",
            epochs="0",
            interference="gaussian",
        )
        reference = r3.per_level(noisy, "input_affine_db")
        block = r3.select(
            matched_rows,
            signal=signal,
            operating_point="controlled",
            interference="gaussian",
        )
        by_level: defaultdict[float, list[r3.Row]] = defaultdict(list)
        for row in block:
            by_level[fnum(row["calibrate_at_snr_db"])].append(row)
        medians[signal] = {}
        for level, group in sorted(by_level.items()):
            ceiling = r3.unit_values(group, "snr_affine_db")
            predictions = [
                r3.law_prediction(reference[level][unit], ceiling[unit])
                for unit in ceiling
                if unit in reference.get(level, {})
            ]
            medians[signal][level] = r3.median(predictions)
    return medians


def curve_panel(
    ax: Axes,
    signals: tuple[str, ...],
    curve: list[dict[str, str]],
    law: dict[str, dict[float, float]],
) -> None:
    first = [
        r
        for r in curve
        if r["signal"] == signals[0]
        and r["arm"] == "controlled-passive"
        and math.isfinite(fnum(r["input_snr_db"]))
    ]
    first.sort(key=lambda r: fnum(r["input_snr_db"]))
    ax.plot(
        [fnum(r["input_snr_db"]) for r in first],
        [fnum(r["input_affine_median"]) for r in first],
        color=REFERENCE,
        linewidth=0.8,
        zorder=1,
    )
    top = 0.0
    for signal in signals:
        color, marker, _ = SIGNALS[signal]
        rows = sorted(
            (
                r
                for r in curve
                if r["signal"] == signal and r["arm"] == "controlled-passive"
            ),
            key=lambda r: fnum(r["input_snr_db"]),
        )
        finite = [r for r in rows if math.isfinite(fnum(r["input_snr_db"]))]
        clean = next(r for r in rows if math.isinf(fnum(r["input_snr_db"])))
        levels = sorted(law[signal])
        ax.plot(
            levels,
            [law[signal][lv] for lv in levels],
            color=color,
            linewidth=0.8,
            zorder=2,
        )
        ax.plot(
            [fnum(r["input_snr_db"]) for r in finite],
            [fnum(r["output_affine_median"]) for r in finite],
            linestyle="none",
            marker=marker,
            markersize=2.6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.7,
            zorder=3,
        )
        median = fnum(clean["output_affine_median"])
        ax.errorbar(
            [CLEAN_X],
            [median],
            yerr=[
                [median - fnum(clean["output_affine_q25"])],
                [fnum(clean["output_affine_q75"]) - median],
            ],
            color=color,
            marker=marker,
            markersize=2.8,
            capsize=1.4,
            elinewidth=0.6,
            zorder=3,
        )
        top = max(top, fnum(clean["output_affine_q75"]))
    ax.axvline(45.0, color="#bdbdbd", linewidth=0.4, zorder=0)
    ax.set_xlim(-23, 54)
    ax.set_ylim(0, top * 1.2 + 0.1)
    ax.set_xticks([-20, 0, 20, 40, CLEAN_X])
    ax.set_xticklabels(["−20", "0", "20", "40", "∞"])
    ax.set_xlabel("Input SNR (dB)")


def figure_1() -> None:
    curve = load(R3_ANALYSIS / "c1_curve.csv")
    levels = load(R3_ANALYSIS / "c1_operating_point_levels.csv")
    law = matched_law_medians()

    fig = plt.figure(figsize=(SINGLE_COLUMN_IN, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0), height_ratios=(1.5, 1.0))
    ax_ecg = fig.add_subplot(grid[0, 0])
    ax_sine = fig.add_subplot(grid[0, 1])
    ax_res = fig.add_subplot(grid[1, :])

    curve_panel(ax_ecg, ("mit-bih", "synthetic-ecg", "eeg"), curve, law)
    curve_panel(ax_sine, ("sine",), curve, law)
    ax_ecg.set_ylabel("Output SNR (dB)")
    ax_ecg.set_title("(a) ECG and EEG", loc="left", pad=2)
    ax_sine.set_title("(b) 2 Hz sine", loc="left", pad=2)

    for signal, (color, marker, _) in SIGNALS.items():
        rows = sorted(
            (
                r
                for r in levels
                if r["signal"] == signal
                and r["arm"] == "controlled-passive"
                and r["interference"] == "gaussian"
            ),
            key=lambda r: fnum(r["input_snr_db"]),
        )
        x = [fnum(r["input_snr_db"]) for r in rows]
        ax_res.plot(
            x,
            [fnum(r["residual_matched_median"]) for r in rows],
            color=color,
            marker=marker,
            markersize=2.4,
            markevery=2,
            zorder=3,
        )
        if signal == "mit-bih":
            ax_res.plot(
                x,
                [fnum(r["residual_ordinary_median"]) for r in rows],
                color=color,
                linestyle=(0, (2.2, 1.6)),
                linewidth=0.8,
                zorder=2,
            )
    ax_res.axhline(0.0, color=INK, linewidth=0.5, zorder=1)
    ax_res.set_xlim(-23, 43)
    ax_res.set_xticks([-20, -10, 0, 10, 20, 30, 40])
    ax_res.set_ylim(-0.75, 0.45)
    ax_res.set_xlabel("Input SNR (dB)")
    ax_res.set_ylabel("Residual (dB)")
    ax_res.set_title("(c) Measured minus cascade law", loc="left", pad=2)
    ax_res.annotate(
        "MIT-BIH, ceiling at the\nnoiseless operating point",
        xy=(6.0, -0.52),
        xytext=(13.0, -0.45),
        fontsize=8,
        color=BLUE,
        va="center",
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.5},
    )

    handles = [
        *[
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=2.8,
                markerfacecolor="white",
                label=label,
            )
            for color, marker, label in SIGNALS.values()
        ],
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        handlelength=1.4,
        columnspacing=0.7,
        handletextpad=0.35,
        fontsize=8,
    )
    save(fig, "fig1_ceiling_law")


# --------------------------------------------------------------------------
# Fig. 2 — fan-in
# --------------------------------------------------------------------------

WIDTHS: dict[str, tuple[str, str]] = {
    "512": ("o", "$N$ = 512, $K$ = 16"),
    "128": ("s", "$N$ = 128, $K$ = 8"),
}


def figure_2() -> None:
    cells = load(R4B_ANALYSIS / "r4b_fan_in_cells.csv")
    training = load(R4B_ANALYSIS / "r4b_training_levels.csv")

    fig, (ax_rate, ax_train) = plt.subplots(
        1,
        2,
        figsize=(SINGLE_COLUMN_IN, 1.85),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.2, 1.0)},
    )
    for width, (marker, label) in WIDTHS.items():
        for sparse in (False, True):
            rows = sorted(
                (
                    r
                    for r in cells
                    if r["width"] == width and (r["fan_in"] != "dense") == sparse
                ),
                key=lambda r: fnum(r["output_rate_hz"]),
            )
            ax_rate.plot(
                [fnum(r["output_rate_hz"]) for r in rows],
                [fnum(r["ceiling_median"]) for r in rows],
                color=GREEN if sparse else REFERENCE,
                linestyle="-" if sparse else (0, (2.2, 1.6)),
                marker=marker,
                markersize=2.8,
                markerfacecolor=GREEN if sparse else "white",
                label=f"{label}" if sparse else None,
            )
    ax_rate.set_xscale("log")
    ax_rate.set_xticks([1, 2, 5, 10, 20, 40])
    ax_rate.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    ax_rate.minorticks_off()
    ax_rate.set_ylim(0, 6.6)
    ax_rate.set_xlabel("Layer firing rate (Hz)")
    ax_rate.set_ylabel("Ceiling (dB)")
    ax_rate.set_title("(a) Fixed fan-in vs dense", loc="left", pad=2)
    dense = Line2D(
        [],
        [],
        color=REFERENCE,
        linestyle=(0, (2.2, 1.6)),
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        label="dense, same $N$",
    )
    handles, _ = ax_rate.get_legend_handles_labels()
    ax_rate.legend(
        handles=[*handles, dense],
        loc="lower right",
        fontsize=8,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.25,
    )

    for signal in ("mit-bih", "synthetic-ecg"):
        color, marker, label = SIGNALS[signal]
        rows = sorted(
            (r for r in training if r["signal"] == signal),
            key=lambda r: fnum(r["epochs"]),
        )
        x = [fnum(r["pr1"]) for r in rows]
        y = [fnum(r["ceiling"]) for r in rows]
        ax_train.plot(
            x, y, color=color, marker=marker, markersize=2.8, label=label, zorder=3
        )
        if signal == "mit-bih":
            for row, xi, yi in zip(rows, x, y, strict=True):
                epochs = row["epochs"]
                if epochs == "50":
                    continue
                if epochs == "200":
                    ax_train.annotate(
                        "50, 200",
                        xy=(xi * 1.02, yi + 0.04),
                        xytext=(1.9, 1.32),
                        fontsize=8,
                        color=INK,
                        va="center",
                        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.5},
                    )
                    continue
                ax_train.annotate(
                    "0 epochs" if epochs == "0" else epochs,
                    xy=(xi, yi),
                    xytext=(0.0, 4.0),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=INK,
                )
    ax_train.set_xscale("log", base=2)
    ax_train.set_xticks([2, 4, 8, 16, 32])
    ax_train.set_xticklabels(["2", "4", "8", "16", "32"])
    ax_train.minorticks_off()
    ax_train.set_xlim(1.7, 40)
    ax_train.set_ylim(0.85, 2.25)
    ax_train.set_xlabel("Layer-1 effective inputs")
    ax_train.set_ylabel("Ceiling (dB)")
    ax_train.set_title("(b) STDP training", loc="left", pad=2)
    ax_train.legend(loc="lower right", fontsize=8, handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig2_fan_in")


# --------------------------------------------------------------------------
# Table I — what moves the MIT-BIH ceiling
# --------------------------------------------------------------------------


def find(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise SystemExit(f"Table I: {len(matches)} rows match {criteria}")
    return matches[0]


def change_cell(row: dict[str, str]) -> str:
    """Median change and BCa interval; bold if significant after Holm."""
    text = (
        f"${fnum(row['median_diff']):+.2f}$ "
        f"[${fnum(row['ci_low']):+.2f}$, ${fnum(row['ci_high']):+.2f}$]"
    )
    if row["significant"] == "True":
        text = f"{{\\boldmath {text}}}"
    if row["equivalent"] == "True":
        text += " $\\equiv$"
    return text


def ceiling_cell(row: dict[str, str], dagger: bool = False) -> str:
    mark = "$^{\\mathrm{a}}$" if dagger else ""
    return (
        f"${fnum(row['reference_median']):.2f}\\,{{\\to}}\\,"
        f"{fnum(row['level_median']):.2f}${mark}"
    )


def table_1() -> None:
    c2 = load(R3_ANALYSIS / "c2_comparisons.csv")
    rate = load(R3_ANALYSIS / "c3_rate_comparisons.csv")
    fan_in = load(R4B_ANALYSIS / "r4b_fan_in_test.csv")
    encoders = load(R5_ANALYSIS / "r5_t1_encoders.csv")

    def knob(
        name: str, level: str, operating_point: str = "controlled"
    ) -> dict[str, str]:
        return find(
            c2,
            knob=name,
            signal="mit-bih",
            input="clean",
            operating_point=operating_point,
            metric="snr_affine_db",
            level=level,
        )

    width = knob("output_width", "64.0")
    width_family = [
        r
        for r in c2
        if r["knob"] == "output_width"
        and r["signal"] == "mit-bih"
        and r["input"] == "clean"
        and r["metric"] == "snr_affine_db"
    ]
    taus = [
        r
        for r in c2
        if r["knob"] == "decoder_tau"
        and r["signal"] == "mit-bih"
        and r["input"] == "clean"
        and r["metric"] == "snr_affine_db"
    ]
    best_tau = max(taus, key=lambda r: fnum(r["median_diff"]))
    temporal = find(encoders, output_rate_hz="40", encoder="temporal", signal="mit-bih")
    delta = find(encoders, output_rate_hz="40", encoder="delta", signal="mit-bih")
    sparse_output = find(
        rate,
        signal="mit-bih",
        scope="output 32-16-64",
        input_snr_db="inf",
        metric="snr_affine_db",
        rate_hz="5.0",
    )
    fan = find(fan_in, width="512")

    # The grouping below is an editorial reading; check it against the data.
    if not all(r["equivalent"] == "True" for r in width_family):
        raise SystemExit("Table I: output width is no longer equivalent")
    if best_tau["equivalent"] != "True" or best_tau["significant"] == "True":
        raise SystemExit("Table I: some readout window lifts the ceiling")
    if temporal["significant"] == "True" and fnum(temporal["median_diff"]) > 0:
        raise SystemExit("Table I: temporal coding raises the ceiling")
    lowers = [
        knob("epochs", "10.0"),
        knob("epochs", "200.0"),
        knob("depth", "4.0"),
        delta,
    ]
    raises = [knob("depth", "2.0"), sparse_output, fan]
    for row in lowers:
        if not (row["significant"] == "True" and fnum(row["median_diff"]) < 0):
            raise SystemExit(f"Table I: not significantly lower: {row}")
    for row in raises:
        if not (row["significant"] == "True" and fnum(row["median_diff"]) > 0):
            raise SystemExit(f"Table I: not significantly higher: {row}")

    tau_ms = int(fnum(best_tau["level"]))
    groups: list[tuple[str, list[tuple[str, str, str]]]] = [
        (
            "Does not raise it",
            [
                ("Output width, 8 to 64", ceiling_cell(width), change_cell(width)),
                (
                    f"Readout $\\tau$, 20 to {tau_ms}~ms (best)",
                    ceiling_cell(best_tau),
                    change_cell(best_tau),
                ),
                (
                    "Time-to-first-spike code",
                    ceiling_cell(temporal, dagger=True),
                    change_cell(temporal),
                ),
            ],
        ),
        (
            "Lowers it",
            [
                (
                    "STDP, 10 epochs",
                    ceiling_cell(lowers[0]),
                    change_cell(lowers[0]),
                ),
                (
                    "STDP, 200 epochs",
                    ceiling_cell(lowers[1]),
                    change_cell(lowers[1]),
                ),
                (
                    "Depth, 3 to 4 layers",
                    ceiling_cell(lowers[2]),
                    change_cell(lowers[2]),
                ),
                (
                    "Delta code",
                    ceiling_cell(delta, dagger=True),
                    change_cell(delta),
                ),
            ],
        ),
        (
            "Raises it",
            [
                (
                    "Depth, 3 to 2 layers",
                    ceiling_cell(raises[0]),
                    change_cell(raises[0]),
                ),
                (
                    "Output rate, 40 to 5~Hz$^{\\mathrm{b}}$",
                    ceiling_cell(sparse_output),
                    change_cell(sparse_output),
                ),
                (
                    f"Fan-in, all to {fan['fan_in']}$^{{\\mathrm{{c}}}}$",
                    ceiling_cell(fan),
                    change_cell(fan),
                ),
            ],
        ),
    ]

    lines = [
        "% Generated by scripts/paper_icee2026/figures.py — do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{What Moves the MIT-BIH Ceiling}",
        "\\label{tab:levers}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lll@{}}",
        "\\toprule",
        "Manipulation & Ceiling (dB) & Change (dB) [95\\% CI] \\\\",
    ]
    for heading, rows in groups:
        lines += [
            "\\midrule",
            f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{heading}}}}} \\\\",
        ]
        lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    notes = (
        "Noiseless-input output SNR, median over 16 records; change is the "
        "paired median with a BCa interval. Reference: untrained $[32,16,8]$, "
        "every layer at 40~Hz, Poisson code, 20~ms exponential readout; "
        "STDP rows hold rates. "
        "$^{\\mathrm{a}}$Delay-compensated SNR against the best Poisson readout "
        "(output advanced by its best lag in 0--64~ms). "
        "$^{\\mathrm{b}}$64 output neurons. "
        "$^{\\mathrm{c}}$One-layer $[512,512]$ network, both at their best rate "
        "(20~Hz). "
        "Bold: significant (Wilcoxon signed-rank test, Holm correction within each "
        "family). "
        "$\\equiv$: equivalent within $\\pm 0.5$~dB (two one-sided tests)."
    )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[2pt]",
        "\\parbox{\\columnwidth}{\\scriptsize " + notes + "}",
        "\\end{table}",
        "",
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "table1_levers.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    plt.rcParams.update(IEEE_RC)
    # The template asks for 8 pt figure labels; R3's style sets ticks and legends at 7.
    plt.rcParams.update(
        {"xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}
    )
    figure_1()
    figure_2()
    table_1()


if __name__ == "__main__":
    main()
