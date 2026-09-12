"""Analysis and plotting utilities for experiment sweeps.

**The unit of analysis is a configuration, not a run.** Phase 3 sweeps 16
MIT-BIH records, and `_build_phase3_tasks` stamps `signal_type="ecg"` on all of
them, so grouping by signal type alone collapsed every record into one bucket
and counted between-patient variance as within-config seed noise. That is
pseudo-replication, and it is visible to any reviewer from the reported *n*.
`dataset` and `record_id` are therefore part of every grouping key here.

For comparisons and invariance claims, prefer `paired_comparison` over reading
`compute_statistical_significance` output directly. It implements the protocol
in `docs/paper_roadmap.md` R2: seed replicates averaged within unit, Wilcoxon
signed-rank on the paired differences, Holm-Bonferroni across the strata of one
family, bootstrap BCa confidence intervals on the median difference, and TOST
equivalence testing against a pre-registered margin.

**A non-significant p-value is not evidence of invariance.** Every "the ceiling
does not change with X" claim is a claim of *no effect* and needs the TOST
verdict, which `paired_comparison` reports alongside the significance one.
"""

import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

# dataset, record_id, signal_type, flicker_freq, input_snr, epochs, stdp_tau
_ConfigKey = tuple[str, str | None, str, float, float, int, float]

# Every column `ResultsDB` writes, excluding the surrogate `id`. Pre-Phase-3
# databases lack the last three; `fetch_raw_runs` fills them from these defaults
# rather than failing, so an archived Phase 2 sweep still reads.
_RUN_COLUMNS: tuple[str, ...] = (
    "signal_type",
    "flicker_freq",
    "input_snr",
    "epochs",
    "stdp_tau",
    "seed",
    "delta_snr",
    "convergence_speed",
    "dataset",
    "record_id",
    "rpeak_integrity",
)

_COLUMN_DEFAULTS: dict[str, Any] = {
    "dataset": "synthetic",
    "record_id": None,
    "rpeak_integrity": None,
}

# Fields that identify a configuration. A comparison holds all of these fixed
# except the one factor under test, and averages over `seed`.
_CONFIG_FIELDS: tuple[str, ...] = (
    "dataset",
    "record_id",
    "signal_type",
    "flicker_freq",
    "input_snr",
    "epochs",
    "stdp_tau",
)


def config_key(row: dict[str, Any]) -> _ConfigKey:
    """The grouping key for one run: everything but the seed and the metrics."""
    record = row.get("record_id")
    return (
        str(row.get("dataset", "synthetic")),
        None if record is None else str(record),
        str(row["signal_type"]),
        float(row["flicker_freq"]),
        float(row["input_snr"]),
        int(row["epochs"]),
        float(row["stdp_tau"]),
    )


def _scipy_stats() -> Any:
    """
    Import `scipy.stats` lazily, returning None when it is unavailable.

    Every statistical function here degrades to a printed message rather than
    raising, so plotting and aggregation still work in an environment without
    scipy. Routing all of them through one import also keeps the single
    `type: ignore` that scipy needs from depending on function order.
    """
    try:
        from scipy import stats  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        return None
    return stats


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
    """
    Fetch every individual run row without aggregation.

    Selects all of `_RUN_COLUMNS`. `dataset`, `record_id` and `rpeak_integrity`
    were added in Phase 3 and were previously dropped here, which made
    `rpeak_integrity` unreachable from the analysis layer entirely and forced
    every record into one group. Columns absent from an older database are
    filled from `_COLUMN_DEFAULTS` so pre-Phase-3 sweeps still read.

    Args:
        db_path: Path to a results database.

    Returns:
        One dict per run, with every key in `_RUN_COLUMNS` present.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    present = {str(row[1]) for row in cursor.execute("PRAGMA table_info(runs)")}
    selected = [column for column in _RUN_COLUMNS if column in present]
    missing = [column for column in _RUN_COLUMNS if column not in present]
    cursor.execute(f"SELECT {', '.join(selected)} FROM runs")  # noqa: S608

    rows: list[dict[str, Any]] = []
    for record in cursor.fetchall():
        row = dict(record)
        for column in missing:
            row[column] = _COLUMN_DEFAULTS.get(column)
        rows.append(row)
    return rows


def compute_statistical_significance(
    raw_runs: list[dict[str, Any]],
    p_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """One-sample t-test per config: H₀ = mean ΔSNR is zero across seeds.

    Returns one result dict per unique configuration — `dataset`, `record_id`,
    `signal_type`, `flicker_freq`, `input_snr`, `epochs`, `stdp_tau` — including
    t-statistic, two-tailed p-value, Cohen's d, and boolean significance /
    improvement flags.

    `dataset` and `record_id` join the key as of R2. Without them a Phase 3
    sweep's 16 records collapsed into one group per parameter combination and
    `n_seeds` reported seeds × records, counting between-patient variance as
    seed noise. Synthetic sweeps are unaffected: every row shares
    `dataset="synthetic"` and `record_id=None`, so the grouping is identical.

    Note:
        This tests whether ΔSNR differs from zero. It cannot support a claim of
        *invariance* — for that use `paired_comparison`, which reports a TOST
        equivalence verdict against a pre-registered margin.
    """
    stats = _scipy_stats()
    if stats is None:
        print("scipy is required for significance testing.")
        return []

    groups: defaultdict[_ConfigKey, list[float]] = defaultdict(list)
    config_meta: dict[_ConfigKey, dict[str, Any]] = {}

    for r in raw_runs:
        key = config_key(r)
        groups[key].append(float(r["delta_snr"]))
        if key not in config_meta:
            config_meta[key] = {
                "dataset": r.get("dataset", "synthetic"),
                "record_id": r.get("record_id"),
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


# ---------------------------------------------------------------------------
# R2 — the statistical protocol
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjustment of a family of p-values.

    Uniformly more powerful than Bonferroni and controls the same family-wise
    error rate. The family is whatever set of tests is reported together, and
    its size must be stated wherever the adjusted values appear.

    Args:
        p_values: Raw p-values. NaN entries are passed through unchanged and
            excluded from the family size, since a test that could not run is
            not a test that was performed.

    Returns:
        Adjusted p-values in the input order, each clipped to at most 1.0.
    """
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    adjusted = [float("nan")] * len(p_values)
    if not indexed:
        return adjusted

    indexed.sort()
    n = len(indexed)
    running = 0.0
    for rank, (p, original) in enumerate(indexed):
        running = max(running, (n - rank) * p)
        adjusted[original] = min(1.0, running)
    return adjusted


def bootstrap_median_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the median, BCa where possible.

    Args:
        values: Sample to resample.
        confidence: Interval coverage.
        n_resamples: Bootstrap resamples.
        seed: Seed for reproducibility — these intervals go in the manuscript.

    Returns:
        `(low, high)`, or `(nan, nan)` when the sample is too small or
        degenerate for an interval to mean anything. A constant sample has zero
        spread, so BCa's acceleration term is undefined; percentile is used as
        the fallback and a fully constant sample returns NaN rather than a
        zero-width interval that would read as infinite precision.
    """
    if len(values) < 2 or len(set(values)) < 2:
        return float("nan"), float("nan")

    import numpy as np  # noqa: PLC0415

    stats = _scipy_stats()
    if stats is None:
        print("scipy is required for bootstrap intervals.")
        return float("nan"), float("nan")

    sample = (np.asarray(values, dtype=float),)
    for method in ("BCa", "percentile"):
        try:
            result = stats.bootstrap(
                sample,
                np.median,
                confidence_level=confidence,
                n_resamples=n_resamples,
                method=method,
                random_state=np.random.default_rng(seed),
            )
        except (ValueError, RuntimeWarning):
            continue
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if math.isfinite(low) and math.isfinite(high):
            return low, high
    return float("nan"), float("nan")


def tost_equivalence(
    differences: list[float],
    margin: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Two one-sided tests for equivalence of paired differences to zero.

    A non-significant difference is not evidence of no difference. TOST inverts
    the burden: the null is that the effect is *at least* `margin`, and
    rejecting it in both directions is positive evidence that the true effect
    lies inside `(-margin, +margin)`.

    Args:
        differences: Paired differences, one per analysis unit.
        margin: Equivalence margin in the metric's units, pre-registered. The
            roadmap proposes ±0.5 dB.
        alpha: Significance level for each one-sided test.

    Returns:
        Dict with `n`, `mean_diff`, `tost_p` (the larger of the two one-sided
        p-values), `equivalent`, and the `(1 - 2*alpha)` interval whose
        containment within the margin is the equivalent verdict.

    Raises:
        ValueError: If `margin` is not strictly positive.
    """
    if margin <= 0.0:
        raise ValueError("Equivalence margin must be strictly positive.")

    n = len(differences)
    base: dict[str, Any] = {
        "n": n,
        "margin": margin,
        "mean_diff": float("nan"),
        "tost_p": float("nan"),
        "equivalent": False,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if n < 2:
        return base

    stats = _scipy_stats()
    if stats is None:
        print("scipy is required for equivalence testing.")
        return base

    mean_diff = sum(differences) / n
    variance = sum((x - mean_diff) ** 2 for x in differences) / (n - 1)
    std_err = math.sqrt(variance / n)
    base["mean_diff"] = mean_diff

    if std_err == 0.0:
        # Every unit moved by exactly the same amount. The effect is known
        # exactly, so equivalence reduces to whether it sits inside the margin.
        base["tost_p"] = 0.0 if abs(mean_diff) < margin else 1.0
        base["equivalent"] = abs(mean_diff) < margin
        base["ci_low"] = base["ci_high"] = mean_diff
        return base

    df = n - 1
    p_lower = float(stats.t.sf((mean_diff + margin) / std_err, df))
    p_upper = float(stats.t.cdf((mean_diff - margin) / std_err, df))
    tost_p = max(p_lower, p_upper)

    half_width = float(stats.t.ppf(1.0 - alpha, df)) * std_err
    base["tost_p"] = tost_p
    base["equivalent"] = tost_p < alpha
    base["ci_low"] = mean_diff - half_width
    base["ci_high"] = mean_diff + half_width
    return base


def _unit_means(
    raw_runs: list[dict[str, Any]],
    factor: str,
    level: Any,
    metric: str,
) -> dict[tuple[Any, ...], float]:
    """Mean `metric` per analysis unit, for runs at one level of `factor`."""
    fields = [f for f in _CONFIG_FIELDS if f != factor]
    buckets: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in raw_runs:
        if row.get(factor) != level:
            continue
        value = row.get(metric)
        if value is None:
            continue
        buckets[tuple(row.get(f) for f in fields)].append(float(value))
    return {key: sum(vals) / len(vals) for key, vals in buckets.items()}


def paired_comparison(
    raw_runs: list[dict[str, Any]],
    factor: str,
    level_a: Any,
    level_b: Any,
    metric: str = "delta_snr",
    stratify_by: str | None = None,
    margin: float = 0.5,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Compare two levels of one factor, paired within analysis unit.

    Implements the R2 protocol. Seed replicates are averaged so the unit of
    analysis is a configuration (a record, for Phase 3 data) rather than a run;
    units are paired across the two levels; Wilcoxon signed-rank tests the
    difference; Holm-Bonferroni corrects across the strata of this one family;
    the median difference gets a bootstrap BCa interval; and TOST reports
    whether the difference is equivalent to zero within `margin`.

    Read `significant` and `equivalent` together. Both false means the data are
    simply uninformative at this sample size — which is the honest reading of
    most non-significant results, and the one a bare p-value hides.

    Args:
        raw_runs: Rows from `fetch_raw_runs`.
        factor: Column varied between the two arms, e.g. `"epochs"`.
        level_a: Baseline level. Differences are `b - a`.
        level_b: Comparison level.
        metric: Column compared, e.g. `"delta_snr"` or `"rpeak_integrity"`.
        stratify_by: Optional config column to split on before testing — the
            roadmap requires stratifying by input-SNR class, because the regime
            advantage is invisible in a global aggregate. Each stratum is one
            test in the family.
        margin: Pre-registered equivalence margin, in the metric's units.
        alpha: Significance level, for both the Wilcoxon and each TOST arm.
        seed: Seed for the bootstrap.

    Returns:
        One dict per stratum (or a single dict when `stratify_by` is None),
        each carrying `n_pairs`, `median_diff`, the bootstrap interval,
        `p_value`, `p_holm`, `significant`, the TOST fields, and
        `family_size` so the correction can be reported honestly.

    Raises:
        ValueError: If `stratify_by` is the factor itself, which would put the
            same runs in both arms of every stratum.
    """
    if stratify_by == factor:
        raise ValueError(f"Cannot stratify by {factor!r}, the factor under comparison.")

    means_a = _unit_means(raw_runs, factor, level_a, metric)
    means_b = _unit_means(raw_runs, factor, level_b, metric)
    fields = [f for f in _CONFIG_FIELDS if f != factor]

    strata: defaultdict[Any, list[tuple[tuple[Any, ...], float]]] = defaultdict(list)
    for key in means_a.keys() & means_b.keys():
        stratum = key[fields.index(stratify_by)] if stratify_by else None
        strata[stratum].append((key, means_b[key] - means_a[key]))

    stats = _scipy_stats()
    if stats is None:
        print("scipy is required for paired comparison.")
        return []

    rows: list[dict[str, Any]] = []
    for stratum in sorted(strata, key=lambda v: (v is not None, v)):
        pairs = strata[stratum]
        diffs = [d for _, d in pairs]
        n = len(diffs)

        statistic, p_value = float("nan"), float("nan")
        if n >= 2 and any(d != 0.0 for d in diffs):
            result = stats.wilcoxon(diffs, zero_method="wilcox")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        low, high = bootstrap_median_ci(diffs, seed=seed)
        equivalence = tost_equivalence(diffs, margin=margin, alpha=alpha)

        row: dict[str, Any] = {
            "factor": factor,
            "level_a": level_a,
            "level_b": level_b,
            "metric": metric,
            "stratify_by": stratify_by,
            "stratum": stratum,
            "n_pairs": n,
            "median_diff": (
                float(sorted(diffs)[n // 2])
                if n % 2
                else (sorted(diffs)[n // 2 - 1] + sorted(diffs)[n // 2]) / 2.0
            )
            if n
            else float("nan"),
            "ci_low": low,
            "ci_high": high,
            "wilcoxon_stat": statistic,
            "p_value": p_value,
        }
        row.update(
            {
                "mean_diff": equivalence["mean_diff"],
                "tost_p": equivalence["tost_p"],
                "equivalent": equivalence["equivalent"],
                "margin": margin,
                "tost_ci_low": equivalence["ci_low"],
                "tost_ci_high": equivalence["ci_high"],
            }
        )
        rows.append(row)

    adjusted = holm_bonferroni([r["p_value"] for r in rows])
    for row, p_holm in zip(rows, adjusted, strict=True):
        row["p_holm"] = p_holm
        row["family_size"] = len(rows)
        row["significant"] = not math.isnan(p_holm) and p_holm < alpha
    return rows
