"""Analysis and plotting utilities for Phase 2 results."""

import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

_ConfigKey = tuple[str, float, float, int, float]


def fetch_aggregated_data(db_path: str | Path) -> list[dict[str, Any]]:
    """Fetch and aggregate runs by averaging over random seeds."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT signal_type, flicker_freq, input_snr, epochs, stdp_tau,
               AVG(delta_snr) as mean_delta_snr,
               AVG(convergence_speed) as mean_convergence
        FROM runs
        GROUP BY signal_type, flicker_freq, input_snr, epochs, stdp_tau
    """)
    return [dict(row) for row in cursor.fetchall()]


def plot_snr_heatmap(
    results: list[dict[str, Any]],
    signal_type: str,
    output_path: str | Path,
) -> None:
    """Generate SNR improvement heatmap (Input SNR x Flicker Freq)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib and numpy are required for plotting.")
        return

    data = [r for r in results if r["signal_type"] == signal_type]
    if not data:
        return

    heatmap_data: defaultdict[float, defaultdict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in data:
        snr = r["input_snr"]
        freq = r["flicker_freq"]
        heatmap_data[snr][freq].append(r["mean_delta_snr"])

    snrs = sorted(set(r["input_snr"] for r in data))
    freqs = sorted(set(r["flicker_freq"] for r in data))

    matrix = np.zeros((len(snrs), len(freqs)))
    for i, s in enumerate(snrs):
        for j, f in enumerate(freqs):
            vals = heatmap_data[s][f]
            matrix[i, j] = float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(freqs)))
    ax.set_yticks(np.arange(len(snrs)))
    ax.set_xticklabels(freqs)
    ax.set_yticklabels(snrs)

    ax.set_xlabel("Flicker Frequency (Hz)")
    ax.set_ylabel("Input SNR (dB)")
    ax.set_title(f"Mean $\\Delta$SNR for {signal_type.upper()}")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("$\\Delta$SNR (dB)")

    for i in range(len(snrs)):
        for j in range(len(freqs)):
            color = "w" if matrix[i, j] < (matrix.max() * 0.7) else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_convergence_curves(
    results: list[dict[str, Any]],
    signal_type: str,
    output_path: str | Path,
) -> None:
    """Generate convergence curves (SNR vs. Epochs)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting.")
        return

    data = [r for r in results if r["signal_type"] == signal_type]
    if not data:
        return

    curves: defaultdict[float, defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in data:
        snr = r["input_snr"]
        epochs = r["epochs"]
        curves[snr][epochs].append(r["mean_delta_snr"])

    snrs = sorted(set(r["input_snr"] for r in data))
    epoch_list = sorted(set(r["epochs"] for r in data))

    fig, ax = plt.subplots(figsize=(8, 6))

    for snr in snrs:
        y = [
            float(sum(curves[snr][e])) / len(curves[snr][e]) if curves[snr][e] else 0.0
            for e in epoch_list
        ]
        ax.plot(epoch_list, y, marker="o", label=f"Input SNR: {snr} dB")

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Mean $\\Delta$SNR (dB)")
    ax.set_title(f"Convergence Curves for {signal_type.upper()}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_parameter_sensitivity(
    results: list[dict[str, Any]],
    signal_type: str,
    output_path: str | Path,
) -> None:
    """Generate parameter sensitivity (STDP tau vs average SNR)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting.")
        return

    data = [r for r in results if r["signal_type"] == signal_type]
    if not data:
        return

    tau_sums: defaultdict[float, float] = defaultdict(float)
    tau_counts: defaultdict[float, int] = defaultdict(int)

    for r in data:
        tau = r["stdp_tau"]
        tau_sums[tau] += r["mean_delta_snr"]
        tau_counts[tau] += 1

    taus = sorted(tau_sums.keys())
    means = [tau_sums[t] / tau_counts[t] for t in taus]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        [str(t) for t in taus],
        means,
        color="skyblue",
        edgecolor="black",
    )

    ax.set_xlabel("STDP $\\tau$ (ms)")
    ax.set_ylabel("Average $\\Delta$SNR (dB)")
    ax.set_title(f"Parameter Sensitivity (STDP $\\tau$) for {signal_type.upper()}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def fetch_raw_runs(db_path: str | Path) -> list[dict[str, Any]]:
    """Fetch every individual run row without aggregation."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT signal_type, flicker_freq, input_snr, epochs, stdp_tau,
               seed, delta_snr, convergence_speed
        FROM runs
    """)
    return [dict(row) for row in cursor.fetchall()]


def compute_statistical_significance(
    raw_runs: list[dict[str, Any]],
    p_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """One-sample t-test per config: H₀ = mean ΔSNR is zero across seeds.

    Returns one result dict per unique (signal_type, flicker_freq, input_snr,
    epochs, stdp_tau) combination, including t-statistic, two-tailed p-value,
    Cohen's d, and boolean significance / improvement flags.
    """
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError:
        print("scipy is required for significance testing.")
        return []

    groups: defaultdict[_ConfigKey, list[float]] = defaultdict(list)
    config_meta: dict[_ConfigKey, dict[str, Any]] = {}

    for r in raw_runs:
        key: _ConfigKey = (
            str(r["signal_type"]),
            float(r["flicker_freq"]),
            float(r["input_snr"]),
            int(r["epochs"]),
            float(r["stdp_tau"]),
        )
        groups[key].append(float(r["delta_snr"]))
        if key not in config_meta:
            config_meta[key] = {
                "signal_type": r["signal_type"],
                "flicker_freq": r["flicker_freq"],
                "input_snr": r["input_snr"],
                "epochs": r["epochs"],
                "stdp_tau": r["stdp_tau"],
            }

    results: list[dict[str, Any]] = []
    for key, delta_snrs in groups.items():
        n = len(delta_snrs)
        mean_val = sum(delta_snrs) / n

        if n < 2:
            t_stat = float("nan")
            p_value = float("nan")
            std_val = 0.0
            cohens_d = float("nan")
        else:
            variance = sum((x - mean_val) ** 2 for x in delta_snrs) / (n - 1)
            std_val = math.sqrt(variance)
            t_result = stats.ttest_1samp(delta_snrs, popmean=0.0)
            t_stat = float(t_result.statistic)
            p_value = float(t_result.pvalue)
            cohens_d = mean_val / std_val if std_val > 0.0 else float("nan")

        entry = dict(config_meta[key])
        entry.update(
            {
                "n_seeds": n,
                "mean_delta_snr": mean_val,
                "std_delta_snr": std_val,
                "t_stat": t_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "significant": (
                    not math.isnan(p_value) and p_value < p_threshold and mean_val > 0.0
                ),
                "improved": mean_val > 0.0,
            }
        )
        results.append(entry)

    return results


def summarize_significance(
    sig_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate significance results into high-level metrics."""
    if not sig_results:
        return {
            "total_configs": 0,
            "significant_configs": 0,
            "improved_configs": 0,
            "pct_significant": 0.0,
            "pct_improved": 0.0,
            "mean_delta_snr_overall": 0.0,
            "best_config": None,
            "worst_config": None,
        }

    total = len(sig_results)
    significant = sum(1 for r in sig_results if r["significant"])
    improved = sum(1 for r in sig_results if r["improved"])
    overall_mean = sum(r["mean_delta_snr"] for r in sig_results) / total
    best = max(sig_results, key=lambda r: r["mean_delta_snr"])
    worst = min(sig_results, key=lambda r: r["mean_delta_snr"])

    return {
        "total_configs": total,
        "significant_configs": significant,
        "improved_configs": improved,
        "pct_significant": 100.0 * significant / total,
        "pct_improved": 100.0 * improved / total,
        "mean_delta_snr_overall": overall_mean,
        "best_config": best,
        "worst_config": worst,
    }


def identify_failure_modes(
    sig_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return configs that exhibit at least one failure mode.

    Failure modes (non-exclusive):
      negative_improvement — mean ΔSNR < 0 (SNN degraded the signal)
      not_significant      — improvement > 0 but p-value above threshold
      high_variance        — std_delta_snr > |mean_delta_snr| (unreliable)
    """
    failures: list[dict[str, Any]] = []
    for r in sig_results:
        reasons: list[str] = []

        if r["mean_delta_snr"] < 0.0:
            reasons.append("negative_improvement")
        elif not r["significant"]:
            reasons.append("not_significant")

        std = float(r["std_delta_snr"])
        mean_abs = abs(float(r["mean_delta_snr"]))
        if std > 0.0 and not math.isnan(std) and std > mean_abs:
            reasons.append("high_variance")

        if reasons:
            failures.append(
                {
                    "signal_type": r["signal_type"],
                    "flicker_freq": r["flicker_freq"],
                    "input_snr": r["input_snr"],
                    "epochs": r["epochs"],
                    "stdp_tau": r["stdp_tau"],
                    "mean_delta_snr": r["mean_delta_snr"],
                    "std_delta_snr": r["std_delta_snr"],
                    "p_value": r["p_value"],
                    "failure_reasons": ";".join(reasons),
                }
            )

    return failures
