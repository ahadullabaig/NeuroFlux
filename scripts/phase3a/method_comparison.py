"""Phase 3A — aggregate the four-method sweep into figures and a CSV.

Loads every row from ``results/phase3_full_sweep.db`` (which the Phase 3A
sweep populated with four methods: snn_trained, snn_passive, butterworth_notch,
lms) and produces:

1. ``results/phase3a/method_aggregates.csv`` — per-(method, input_snr) mean/std
   of delta_snr and rpeak_integrity. The threshold-pinning step in
   ``tests/baselines/test_phase3a_regression.py`` reads from this CSV.

2. ``results/phase3a/method_matrix.png`` — 2×4 grid: top row delta_snr by
   method per input-SNR class; bottom row rpeak_integrity. Each cell shows
   the mean across all (record, freq, seed) combinations for that
   (method, input_snr) cell.

3. ``results/phase3a/dual_channel_rpeak.png`` — the headline figure for the
   pivot. Side-by-side violin/box of rpeak_integrity at each input SNR,
   showing snn_passive's rpeak distribution vs the classical methods.
   This is the measurement gate A hinges on.

Run::

    uv run python scripts/phase3a/method_comparison.py
"""

from __future__ import annotations

import csv
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DB_PATH = Path("results/phase3_full_sweep.db")
OUT_DIR = Path("results/phase3a")
METHODS = ["butterworth_notch", "lms", "snn_passive", "snn_trained"]
METHOD_COLORS = {
    "butterworth_notch": "#2c7fb8",
    "lms": "#7fcdbb",
    "snn_passive": "#d95f0e",
    "snn_trained": "#993404",
}
METHOD_LABELS = {
    "butterworth_notch": "Butterworth\nnotch",
    "lms": "LMS",
    "snn_passive": "SNN\n(passive)",
    "snn_trained": "SNN\n(trained)",
}
INPUT_SNRS = [0.0, 5.0, 10.0, 20.0]


def _fetch_rows(db_path: Path) -> list[dict[str, float | str]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT method, input_snr, delta_snr, rpeak_integrity, "
        "record_id, flicker_freq, seed, epochs, stdp_tau "
        "FROM runs WHERE dataset = 'mit-bih'"
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _aggregate(
    rows: list[dict[str, float | str]],
) -> dict[tuple[str, float], dict[str, float]]:
    buckets: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(
        lambda: {"delta_snr": [], "rpeak": []}
    )
    for r in rows:
        key = (str(r["method"]), float(r["input_snr"]))
        buckets[key]["delta_snr"].append(float(r["delta_snr"]))
        if r["rpeak_integrity"] is not None:
            buckets[key]["rpeak"].append(float(r["rpeak_integrity"]))

    agg: dict[tuple[str, float], dict[str, float]] = {}
    for key, vals in buckets.items():
        agg[key] = {
            "n": float(len(vals["delta_snr"])),
            "delta_snr_mean": statistics.mean(vals["delta_snr"]),
            "delta_snr_std": (
                statistics.stdev(vals["delta_snr"])
                if len(vals["delta_snr"]) > 1
                else 0.0
            ),
            "rpeak_mean": statistics.mean(vals["rpeak"])
            if vals["rpeak"]
            else float("nan"),
            "rpeak_std": (
                statistics.stdev(vals["rpeak"]) if len(vals["rpeak"]) > 1 else 0.0
            ),
        }
    return agg


def _write_csv(agg: dict[tuple[str, float], dict[str, float]], out: Path) -> None:
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "input_snr",
                "n",
                "delta_snr_mean",
                "delta_snr_std",
                "rpeak_mean",
                "rpeak_std",
            ]
        )
        for method in METHODS:
            for snr in INPUT_SNRS:
                key = (method, snr)
                if key not in agg:
                    continue
                v = agg[key]
                writer.writerow(
                    [
                        method,
                        snr,
                        int(v["n"]),
                        f"{v['delta_snr_mean']:.4f}",
                        f"{v['delta_snr_std']:.4f}",
                        f"{v['rpeak_mean']:.4f}",
                        f"{v['rpeak_std']:.4f}",
                    ]
                )


def _print_table(agg: dict[tuple[str, float], dict[str, float]]) -> None:
    print("\n=== Phase 3A — method × input_snr aggregates ===\n")
    header = f"{'method':<20}" + "".join(f"{snr:>6.0f} dB " for snr in INPUT_SNRS)
    print(header)
    print("-" * len(header))
    for metric_name, key in [
        ("ΔSNR  (mean)", "delta_snr_mean"),
        ("rpeak (mean)", "rpeak_mean"),
    ]:
        print(f"  {metric_name}")
        for method in METHODS:
            row = f"    {method:<18}"
            for snr in INPUT_SNRS:
                v = agg.get((method, snr))
                row += f"{v[key]:>7.3f}  " if v else f"{'--':>9}"
            print(row)
        print()


def _plot_method_matrix(
    agg: dict[tuple[str, float], dict[str, float]], out: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    x = list(range(len(INPUT_SNRS)))
    bar_width = 0.20

    for i, method in enumerate(METHODS):
        deltas = [
            agg.get((method, snr), {}).get("delta_snr_mean", 0.0) for snr in INPUT_SNRS
        ]
        delta_err = [
            agg.get((method, snr), {}).get("delta_snr_std", 0.0) for snr in INPUT_SNRS
        ]
        rpeaks = [
            agg.get((method, snr), {}).get("rpeak_mean", 0.0) for snr in INPUT_SNRS
        ]
        rpeak_err = [
            agg.get((method, snr), {}).get("rpeak_std", 0.0) for snr in INPUT_SNRS
        ]
        offsets = [xi + (i - 1.5) * bar_width for xi in x]
        axes[0].bar(
            offsets,
            deltas,
            bar_width,
            yerr=delta_err,
            label=METHOD_LABELS[method].replace("\n", " "),
            color=METHOD_COLORS[method],
            capsize=2,
        )
        axes[1].bar(
            offsets,
            rpeaks,
            bar_width,
            yerr=rpeak_err,
            color=METHOD_COLORS[method],
            capsize=2,
        )

    axes[0].axhline(0, color="#999", linewidth=0.8)
    axes[0].set_ylabel("ΔSNR (dB)")
    axes[0].set_title(
        "Phase 3A — four-method comparison on MIT-BIH (16 records × 2 freqs × 5 seeds)"
    )
    axes[0].legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
    axes[1].set_ylabel("R-peak integrity")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Input SNR (dB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(s)}" for s in INPUT_SNRS])

    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_dual_channel(rows: list[dict[str, float | str]], out: Path) -> None:
    """Headline figure: rpeak distribution per method at each input SNR."""
    fig, axes = plt.subplots(1, len(INPUT_SNRS), figsize=(13, 4.0), sharey=True)
    for ax, snr in zip(axes, INPUT_SNRS, strict=True):
        data: list[list[float]] = []
        for method in METHODS:
            vals = [
                float(r["rpeak_integrity"])
                for r in rows
                if r["method"] == method
                and float(r["input_snr"]) == snr
                and r["rpeak_integrity"] is not None
            ]
            data.append(vals)

        positions = list(range(len(METHODS)))
        # box plot per method at this SNR
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, method in zip(bp["boxes"], METHODS, strict=True):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.85)

        ax.set_title(f"Input SNR = {int(snr)} dB", fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("R-peak integrity")
    fig.suptitle(
        "Phase 3A — R-peak preservation per method per input-SNR class\n"
        "(dual-channel pitch hinges on passive-SNN bar vs classical bars at low SNR)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(
            f"Database not found at {DB_PATH}. Run the Phase 3A sweep first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _fetch_rows(DB_PATH)
    print(f"Loaded {len(rows)} mit-bih rows from {DB_PATH}.")

    agg = _aggregate(rows)
    _print_table(agg)

    csv_path = OUT_DIR / "method_aggregates.csv"
    _write_csv(agg, csv_path)
    print(f"Wrote per-cell aggregates → {csv_path}")

    matrix_path = OUT_DIR / "method_matrix.png"
    _plot_method_matrix(agg, matrix_path)
    print(f"Wrote method matrix → {matrix_path}")

    dual_path = OUT_DIR / "dual_channel_rpeak.png"
    _plot_dual_channel(rows, dual_path)
    print(f"Wrote dual-channel rpeak figure → {dual_path}")


if __name__ == "__main__":
    main()
