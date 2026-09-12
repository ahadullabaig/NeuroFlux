"""Tests for the R2 statistical protocol.

Two things are being defended here.

The first is the **unit of analysis**. A Phase 3 sweep runs 16 MIT-BIH records
and stamps `signal_type="ecg"` on every one, so a grouping key without
`record_id` collapses them all together and reports `n_seeds` as seeds ×
records. `test_record_id_is_part_of_the_grouping_key` fails if that regresses,
which `docs/paper_roadmap.md` R2 requires explicitly.

The second is that **a non-significant result is not invariance**. The tests
around `tost_equivalence` and `paired_comparison` pin the distinction: data can
be significantly different, equivalent within a margin, or simply
uninformative, and the three are reported separately.
"""

import math
import random
from typing import Any

import pytest
from neuroflux_datasets.analysis import (
    bootstrap_median_ci,
    compute_statistical_significance,
    config_key,
    holm_bonferroni,
    paired_comparison,
    tost_equivalence,
)


def _run(**overrides: Any) -> dict[str, Any]:
    """One run row with sweep defaults, overridden as needed."""
    row: dict[str, Any] = {
        "dataset": "synthetic",
        "record_id": None,
        "signal_type": "ecg",
        "flicker_freq": 50.0,
        "input_snr": 0.0,
        "epochs": 10,
        "stdp_tau": 20.0,
        "seed": 0,
        "delta_snr": 1.0,
        "convergence_speed": 5,
        "rpeak_integrity": 1.0,
    }
    row.update(overrides)
    return row


def _phase3_runs(
    records: list[str], seeds: int, delta_by_record: dict[str, float]
) -> list[dict[str, Any]]:
    """A Phase 3-shaped sweep: many records, each with several seeds."""
    return [
        _run(
            dataset="mit-bih",
            record_id=record,
            seed=seed,
            delta_snr=delta_by_record[record] + 0.01 * seed,
        )
        for record in records
        for seed in range(seeds)
    ]


# ---------------------------------------------------------------------------
# the unit of analysis
# ---------------------------------------------------------------------------


def test_record_id_is_part_of_the_grouping_key() -> None:
    """
    The pseudo-replication regression.

    Sixteen records at five seeds must produce sixteen groups of five, not one
    group of eighty. Dropping `record_id` from `config_key` fails this.
    """
    records = [f"1{i:02d}" for i in range(16)]
    runs = _phase3_runs(records, seeds=5, delta_by_record=dict.fromkeys(records, 2.0))

    results = compute_statistical_significance(runs)

    assert len(results) == 16
    assert all(r["n_seeds"] == 5 for r in results)
    assert {r["record_id"] for r in results} == set(records)


def test_group_count_is_the_product_of_the_swept_factors() -> None:
    """Group count must equal records × flicker × snr × epochs × tau."""
    runs = [
        _run(
            dataset="mit-bih",
            record_id=record,
            flicker_freq=freq,
            input_snr=snr,
            epochs=epochs,
            seed=seed,
        )
        for record in ("100", "214")
        for freq in (50.0, 60.0)
        for snr in (0.0, 10.0)
        for epochs in (0, 10)
        for seed in range(3)
    ]
    results = compute_statistical_significance(runs)
    assert len(results) == 2 * 2 * 2 * 2
    assert all(r["n_seeds"] == 3 for r in results)


def test_dataset_separates_synthetic_from_real() -> None:
    """Same parameters on different datasets are different configurations."""
    runs = [_run(dataset="synthetic"), _run(dataset="mit-bih", record_id="214")]
    assert len({config_key(r) for r in runs}) == 2


def test_synthetic_grouping_is_unchanged() -> None:
    """
    Phase 2 behaviour must be byte-identical.

    Every synthetic row shares `dataset="synthetic"` and `record_id=None`, so
    adding them to the key cannot split anything.
    """
    runs = [_run(seed=s, delta_snr=1.0 + 0.1 * s) for s in range(5)]
    results = compute_statistical_significance(runs)
    assert len(results) == 1
    assert results[0]["n_seeds"] == 5


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def test_holm_is_identity_on_a_single_test() -> None:
    assert holm_bonferroni([0.03]) == [pytest.approx(0.03)]


def test_holm_adjusts_smallest_by_family_size() -> None:
    adjusted = holm_bonferroni([0.01, 0.02, 0.03])
    assert adjusted[0] == pytest.approx(0.03)


def test_holm_is_monotone_and_never_exceeds_one() -> None:
    raw = [0.001, 0.04, 0.2, 0.5, 0.9]
    adjusted = holm_bonferroni(raw)
    assert all(a <= b + 1e-12 for a, b in zip(adjusted, adjusted[1:], strict=False))
    assert all(0.0 <= a <= 1.0 for a in adjusted)


def test_holm_is_never_more_lenient_than_raw() -> None:
    raw = [0.001, 0.04, 0.2]
    assert all(a >= r for a, r in zip(holm_bonferroni(raw), raw, strict=True))


def test_holm_is_less_conservative_than_bonferroni() -> None:
    """The reason to prefer Holm: same error control, more power."""
    raw = [0.01, 0.02, 0.03]
    adjusted = holm_bonferroni(raw)
    bonferroni = [min(1.0, 3 * p) for p in raw]
    assert all(a <= b + 1e-12 for a, b in zip(adjusted, bonferroni, strict=True))
    assert adjusted[2] < bonferroni[2]


def test_holm_preserves_input_order() -> None:
    assert holm_bonferroni([0.5, 0.001])[1] < holm_bonferroni([0.5, 0.001])[0]


def test_holm_passes_nan_through_and_shrinks_the_family() -> None:
    """A test that could not run is not a test that was performed."""
    adjusted = holm_bonferroni([0.01, float("nan"), 0.02])
    assert math.isnan(adjusted[1])
    assert adjusted[0] == pytest.approx(0.02)


def test_holm_on_empty_family() -> None:
    assert holm_bonferroni([]) == []


# ---------------------------------------------------------------------------
# equivalence
# ---------------------------------------------------------------------------


def test_tost_accepts_a_tight_cluster_near_zero() -> None:
    diffs = [0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.02, -0.01]
    result = tost_equivalence(diffs, margin=0.5)
    assert result["equivalent"] is True
    assert result["tost_p"] < 0.05
    assert -0.5 < result["ci_low"] and result["ci_high"] < 0.5


def test_tost_rejects_a_real_effect() -> None:
    diffs = [2.0, 2.1, 1.9, 2.2, 1.8, 2.0, 2.1]
    result = tost_equivalence(diffs, margin=0.5)
    assert result["equivalent"] is False


def test_tost_rejects_noisy_data_centred_on_zero() -> None:
    """
    The case a bare p-value gets wrong.

    These differences average near zero but scatter far past the margin, so
    there is no evidence of equivalence — only an absence of evidence of a
    difference.
    """
    diffs = [3.0, -3.1, 2.8, -2.9, 3.2, -3.0]
    result = tost_equivalence(diffs, margin=0.5)
    assert result["equivalent"] is False
    assert abs(result["mean_diff"]) < 0.5


def test_tost_needs_more_data_for_a_tighter_margin() -> None:
    diffs = [0.2, 0.25, 0.15, 0.22, 0.18, 0.2, 0.21, 0.19]
    assert tost_equivalence(diffs, margin=0.5)["equivalent"] is True
    assert tost_equivalence(diffs, margin=0.1)["equivalent"] is False


def test_tost_handles_a_constant_difference() -> None:
    """Zero spread means the effect is known exactly, not that it is absent."""
    assert tost_equivalence([0.2] * 5, margin=0.5)["equivalent"] is True
    assert tost_equivalence([0.9] * 5, margin=0.5)["equivalent"] is False


def test_tost_declines_on_too_few_pairs() -> None:
    result = tost_equivalence([0.1], margin=0.5)
    assert result["equivalent"] is False
    assert math.isnan(result["tost_p"])


def test_tost_rejects_a_non_positive_margin() -> None:
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="strictly positive"):
            tost_equivalence([0.1, 0.2], margin=bad)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_median() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.15, 0.95, 1.05]
    low, high = bootstrap_median_ci(values, seed=0)
    assert low <= sorted(values)[len(values) // 2] <= high


def test_bootstrap_ci_is_reproducible() -> None:
    """These intervals go in the manuscript, so the seed must pin them."""
    values = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0]
    assert bootstrap_median_ci(values, seed=7) == bootstrap_median_ci(values, seed=7)


def _ci_width(interval: tuple[float, float]) -> float:
    return interval[1] - interval[0]


def test_bootstrap_ci_narrows_with_more_data() -> None:
    rng = random.Random(3)
    small = [rng.gauss(1.0, 0.3) for _ in range(8)]
    large = [rng.gauss(1.0, 0.3) for _ in range(200)]
    assert _ci_width(bootstrap_median_ci(large, seed=1)) < _ci_width(
        bootstrap_median_ci(small, seed=1)
    )


def test_bootstrap_falls_back_when_bca_cannot_be_computed() -> None:
    """
    BCa's acceleration term is undefined on a near-degenerate sample.

    Repeating five values gives a jackknife with no spread, so BCa declines and
    the percentile method has to carry it. A NaN here would mean the fallback
    stopped working.
    """
    low, high = bootstrap_median_ci([1.0, 1.4, 0.6, 1.2, 0.8] * 20, seed=1)
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= 1.0 <= high


def test_bootstrap_ci_declines_on_degenerate_input() -> None:
    """A constant sample has no spread; a zero-width CI would read as certainty."""
    for values in ([], [1.0], [2.0] * 10):
        low, high = bootstrap_median_ci(values)
        assert math.isnan(low) and math.isnan(high)


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def _epoch_runs(
    records: list[str], effect: dict[str, float], snrs: tuple[float, ...] = (0.0,)
) -> list[dict[str, Any]]:
    """Runs at epochs 0 and 10 for each record, with a per-SNR effect."""
    return [
        _run(
            dataset="mit-bih",
            record_id=record,
            input_snr=snr,
            epochs=epochs,
            seed=seed,
            delta_snr=(1.0 + 0.05 * seed + (effect[str(snr)] if epochs else 0.0)),
        )
        for record in records
        for snr in snrs
        for epochs in (0, 10)
        for seed in range(3)
    ]


def test_pairs_are_records_not_runs() -> None:
    """Seeds are averaged within unit, so n_pairs counts records."""
    records = [f"2{i:02d}" for i in range(8)]
    rows = paired_comparison(
        _epoch_runs(records, {"0.0": 1.5}), factor="epochs", level_a=0, level_b=10
    )
    assert len(rows) == 1
    assert rows[0]["n_pairs"] == 8


def test_detects_a_real_difference() -> None:
    records = [f"2{i:02d}" for i in range(10)]
    row = paired_comparison(
        _epoch_runs(records, {"0.0": 1.5}), factor="epochs", level_a=0, level_b=10
    )[0]
    assert row["significant"] is True
    assert row["median_diff"] == pytest.approx(1.5, abs=0.01)
    assert row["equivalent"] is False


def test_reports_equivalence_when_the_effect_is_inside_the_margin() -> None:
    records = [f"2{i:02d}" for i in range(10)]
    row = paired_comparison(
        _epoch_runs(records, {"0.0": 0.02}),
        factor="epochs",
        level_a=0,
        level_b=10,
        margin=0.5,
    )[0]
    assert row["equivalent"] is True
    assert abs(row["median_diff"]) < 0.5


def test_stratification_splits_the_family_and_corrects_across_it() -> None:
    """The regime advantage is invisible in a global aggregate."""
    records = [f"2{i:02d}" for i in range(10)]
    rows = paired_comparison(
        _epoch_runs(records, {"0.0": 2.0, "20.0": 0.01}, snrs=(0.0, 20.0)),
        factor="epochs",
        level_a=0,
        level_b=10,
        stratify_by="input_snr",
    )
    assert len(rows) == 2
    assert all(r["family_size"] == 2 for r in rows)
    by_snr = {r["stratum"]: r for r in rows}
    assert by_snr[0.0]["significant"] is True
    assert by_snr[20.0]["equivalent"] is True
    assert all(r["p_holm"] >= r["p_value"] for r in rows)


def test_unpaired_units_are_dropped() -> None:
    """A record present at only one level cannot contribute a difference."""
    runs = _epoch_runs(["100", "101"], {"0.0": 1.0})
    runs.append(_run(dataset="mit-bih", record_id="999", epochs=0))
    row = paired_comparison(runs, factor="epochs", level_a=0, level_b=10)[0]
    assert row["n_pairs"] == 2


def test_differences_are_b_minus_a() -> None:
    records = [f"2{i:02d}" for i in range(6)]
    runs = _epoch_runs(records, {"0.0": 1.0})
    forward = paired_comparison(runs, factor="epochs", level_a=0, level_b=10)[0]
    reverse = paired_comparison(runs, factor="epochs", level_a=10, level_b=0)[0]
    assert forward["median_diff"] == pytest.approx(-reverse["median_diff"])


def test_other_metrics_are_comparable() -> None:
    """`rpeak_integrity` was unreachable from this layer before R2."""
    runs = [
        _run(
            dataset="mit-bih",
            record_id=record,
            epochs=epochs,
            seed=seed,
            rpeak_integrity=1.0 if epochs == 0 else 0.6,
        )
        for record in ("100", "101", "103", "105", "108", "112")
        for epochs in (0, 10)
        for seed in range(3)
    ]
    row = paired_comparison(
        runs, factor="epochs", level_a=0, level_b=10, metric="rpeak_integrity"
    )[0]
    assert row["median_diff"] == pytest.approx(-0.4)


def test_rejects_stratifying_by_the_factor_itself() -> None:
    with pytest.raises(ValueError, match="factor under comparison"):
        paired_comparison(
            _epoch_runs(["100"], {"0.0": 1.0}),
            factor="epochs",
            level_a=0,
            level_b=10,
            stratify_by="epochs",
        )


def test_no_overlapping_units_gives_no_rows() -> None:
    rows = paired_comparison(
        _epoch_runs(["100"], {"0.0": 1.0}), factor="epochs", level_a=0, level_b=999
    )
    assert rows == []
