"""Comprehensive tests for analysis and plotting utilities (Phase 2)."""

import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from neuroflux_datasets.analysis import (
    compute_statistical_significance,
    fetch_aggregated_data,
    fetch_raw_runs,
    identify_failure_modes,
    plot_convergence_curves,
    plot_parameter_sensitivity,
    plot_snr_heatmap,
    summarize_significance,
)

# ---------------------------------------------------------------------------
# Helpers: create test databases with known data
# ---------------------------------------------------------------------------


def _create_test_db(db_path: Path, rows: list[tuple[Any, ...]]) -> None:
    """Create a SQLite database with the runs table and insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT,
            flicker_freq REAL,
            input_snr REAL,
            epochs INTEGER,
            stdp_tau REAL,
            seed INTEGER,
            delta_snr REAL,
            convergence_speed INTEGER
        )
    """)
    conn.executemany(
        """
        INSERT INTO runs
        (signal_type, flicker_freq, input_snr, epochs, stdp_tau,
         seed, delta_snr, convergence_speed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _sample_rows() -> list[tuple[Any, ...]]:
    """Generate a small set of test rows with known aggregation results.

    Config: ecg, 50Hz, 10dB SNR, 50 epochs, tau=20.0
    Seeds 0,1,2 with delta_snr = 4.0, 6.0, 8.0
    → Expected AVG(delta_snr) = 6.0, AVG(convergence_speed) = 20.0
    """
    return [
        # (signal_type, flicker_freq, input_snr, epochs, stdp_tau,
        #  seed, delta_snr, convergence_speed)
        ("ecg", 50.0, 10.0, 50, 20.0, 0, 4.0, 15),
        ("ecg", 50.0, 10.0, 50, 20.0, 1, 6.0, 20),
        ("ecg", 50.0, 10.0, 50, 20.0, 2, 8.0, 25),
    ]


def _multi_config_rows() -> list[tuple[Any, ...]]:
    """Generate rows spanning multiple signal types, freqs, SNRs, taus."""
    rows = []
    for sig in ["ecg", "eeg"]:
        for freq in [50.0, 60.0]:
            for snr in [5.0, 10.0]:
                for epochs in [50, 100]:
                    for tau in [10.0, 20.0]:
                        for seed in range(2):
                            delta = (
                                2.0
                                + snr * 0.1
                                + tau * 0.05
                                + epochs * 0.01
                                + seed * 0.5
                            )
                            rows.append(
                                (
                                    sig,
                                    freq,
                                    snr,
                                    epochs,
                                    tau,
                                    seed,
                                    delta,
                                    epochs // 2,
                                )
                            )
    return rows


# ---------------------------------------------------------------------------
# Step 1: Tests for fetch_aggregated_data
# ---------------------------------------------------------------------------


class TestFetchAggregatedData:
    """Tests for fetch_aggregated_data()."""

    def test_basic_aggregation(self, tmp_path: Path) -> None:
        """AVG over seeds produces correct mean values."""
        db_path = tmp_path / "test.db"
        _create_test_db(db_path, _sample_rows())

        results = fetch_aggregated_data(db_path)

        assert len(results) == 1  # 1 unique config, 3 seeds
        row = results[0]
        assert row["signal_type"] == "ecg"
        assert row["flicker_freq"] == 50.0
        assert row["input_snr"] == 10.0
        assert row["epochs"] == 50
        assert row["stdp_tau"] == 20.0
        assert abs(row["mean_delta_snr"] - 6.0) < 1e-10
        assert abs(row["mean_convergence"] - 20.0) < 1e-10

    def test_grouping_by_config(self, tmp_path: Path) -> None:
        """Different configs produce separate aggregated rows."""
        db_path = tmp_path / "test.db"
        rows = [
            ("ecg", 50.0, 10.0, 50, 20.0, 0, 4.0, 10),
            ("ecg", 50.0, 10.0, 50, 20.0, 1, 6.0, 20),
            ("eeg", 60.0, 5.0, 100, 10.0, 0, 3.0, 30),
            ("eeg", 60.0, 5.0, 100, 10.0, 1, 5.0, 40),
        ]
        _create_test_db(db_path, rows)

        results = fetch_aggregated_data(db_path)
        assert len(results) == 2

        # Sort by signal_type for stable assertions
        results.sort(key=lambda r: r["signal_type"])

        assert results[0]["signal_type"] == "ecg"
        assert abs(results[0]["mean_delta_snr"] - 5.0) < 1e-10

        assert results[1]["signal_type"] == "eeg"
        assert abs(results[1]["mean_delta_snr"] - 4.0) < 1e-10

    def test_empty_database(self, tmp_path: Path) -> None:
        """Empty database returns empty list."""
        db_path = tmp_path / "empty.db"
        _create_test_db(db_path, [])

        results = fetch_aggregated_data(db_path)
        assert results == []

    def test_single_seed_no_averaging(self, tmp_path: Path) -> None:
        """With only 1 seed, AVG returns the single value."""
        db_path = tmp_path / "single.db"
        rows = [("ecg", 50.0, 10.0, 50, 20.0, 0, 7.5, 12)]
        _create_test_db(db_path, rows)

        results = fetch_aggregated_data(db_path)
        assert len(results) == 1
        assert abs(results[0]["mean_delta_snr"] - 7.5) < 1e-10
        assert abs(results[0]["mean_convergence"] - 12.0) < 1e-10

    def test_multi_config_count(self, tmp_path: Path) -> None:
        """Multi-config dataset produces correct number of groups."""
        db_path = tmp_path / "multi.db"
        rows = _multi_config_rows()
        _create_test_db(db_path, rows)

        results = fetch_aggregated_data(db_path)
        # 2 signals × 2 freqs × 2 SNRs × 2 epochs × 2 taus = 32 groups
        assert len(results) == 32

    def test_result_keys(self, tmp_path: Path) -> None:
        """Each result dict has the expected keys."""
        db_path = tmp_path / "keys.db"
        _create_test_db(db_path, _sample_rows())

        results = fetch_aggregated_data(db_path)
        expected_keys = {
            "signal_type",
            "flicker_freq",
            "input_snr",
            "epochs",
            "stdp_tau",
            "mean_delta_snr",
            "mean_convergence",
        }
        assert set(results[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# Step 2: Tests for plotting functions
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_results(tmp_path: Path) -> list[dict[str, Any]]:
    """Aggregated results fixture for plot tests."""
    db_path = tmp_path / "plot_data.db"
    _create_test_db(db_path, _multi_config_rows())
    return fetch_aggregated_data(db_path)


class TestPlotSnrHeatmap:
    """Tests for plot_snr_heatmap()."""

    def test_creates_file(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Heatmap is saved to the specified path."""
        out = tmp_path / "heatmap.png"
        plot_snr_heatmap(sample_results, "ecg", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_different_signal_types(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Each signal type produces its own valid plot."""
        for sig in ["ecg", "eeg"]:
            out = tmp_path / f"heatmap_{sig}.png"
            plot_snr_heatmap(sample_results, sig, out)
            assert out.exists()

    def test_empty_data_no_crash(self, tmp_path: Path) -> None:
        """Empty results list does not crash or create a file."""
        out = tmp_path / "empty.png"
        plot_snr_heatmap([], "ecg", out)
        assert not out.exists()

    def test_no_matching_signal_type(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Non-existent signal type returns early without creating file."""
        out = tmp_path / "missing.png"
        plot_snr_heatmap(sample_results, "ppg", out)
        assert not out.exists()

    def test_single_data_point(self, tmp_path: Path) -> None:
        """A single data point still produces a valid plot."""
        results = [
            {
                "signal_type": "ecg",
                "flicker_freq": 50.0,
                "input_snr": 10.0,
                "epochs": 50,
                "stdp_tau": 20.0,
                "mean_delta_snr": 5.0,
                "mean_convergence": 25.0,
            }
        ]
        out = tmp_path / "single.png"
        plot_snr_heatmap(results, "ecg", out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotConvergenceCurves:
    """Tests for plot_convergence_curves()."""

    def test_creates_file(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "convergence.png"
        plot_convergence_curves(sample_results, "ecg", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_data_no_crash(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_convergence_curves([], "ecg", out)
        assert not out.exists()

    def test_no_matching_signal_type(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "missing.png"
        plot_convergence_curves(sample_results, "ppg", out)
        assert not out.exists()

    def test_single_data_point(self, tmp_path: Path) -> None:
        results = [
            {
                "signal_type": "sine",
                "flicker_freq": 50.0,
                "input_snr": 10.0,
                "epochs": 50,
                "stdp_tau": 20.0,
                "mean_delta_snr": 3.0,
                "mean_convergence": 10.0,
            }
        ]
        out = tmp_path / "single.png"
        plot_convergence_curves(results, "sine", out)
        assert out.exists()


class TestPlotParameterSensitivity:
    """Tests for plot_parameter_sensitivity()."""

    def test_creates_file(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "sensitivity.png"
        plot_parameter_sensitivity(sample_results, "eeg", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_data_no_crash(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.png"
        plot_parameter_sensitivity([], "ecg", out)
        assert not out.exists()

    def test_no_matching_signal_type(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "missing.png"
        plot_parameter_sensitivity(sample_results, "ppg", out)
        assert not out.exists()

    def test_single_tau_value(self, tmp_path: Path) -> None:
        results = [
            {
                "signal_type": "ecg",
                "flicker_freq": 50.0,
                "input_snr": 10.0,
                "epochs": 50,
                "stdp_tau": 20.0,
                "mean_delta_snr": 5.0,
                "mean_convergence": 25.0,
            }
        ]
        out = tmp_path / "single_tau.png"
        plot_parameter_sensitivity(results, "ecg", out)
        assert out.exists()

    def test_multiple_taus_creates_bars(
        self,
        sample_results: list[dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """With multiple tau values, plot should still succeed."""
        out = tmp_path / "multi_tau.png"
        plot_parameter_sensitivity(sample_results, "ecg", out)
        assert out.exists()
        # File should be reasonably large (actual plot, not empty)
        assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Step 3: Tests for Group C — fetch_raw_runs, compute_statistical_significance,
#         summarize_significance, identify_failure_modes
# ---------------------------------------------------------------------------

# mean=5.0, std≈0.17, n=10 → t≈92, p≈0 — clearly significant
_CLEARLY_POSITIVE = [4.8, 5.0, 5.2, 4.9, 5.1, 4.7, 5.3, 5.0, 4.8, 5.2]

# mean≈0.25, std≈1.97, n=10 → t≈0.40, p≈0.70 — not significant at α=0.05
_INSIGNIFICANT = [2.0, -1.5, 3.0, -2.0, 1.5, -1.0, 2.5, -1.8, 1.2, -1.4]

# mean=-5.0, std≈0.17, n=5 — clearly negative
_CLEARLY_NEGATIVE = [-4.8, -5.0, -5.2, -4.9, -5.1]


def _make_raw_runs(
    delta_snrs: list[float],
    signal_type: str = "ecg",
    flicker_freq: float = 50.0,
    input_snr: float = 10.0,
    epochs: int = 50,
    stdp_tau: float = 20.0,
) -> list[dict[str, Any]]:
    """Build raw run dicts with known delta_snr values across seeds."""
    return [
        {
            "signal_type": signal_type,
            "flicker_freq": flicker_freq,
            "input_snr": input_snr,
            "epochs": epochs,
            "stdp_tau": stdp_tau,
            "seed": i,
            "delta_snr": v,
            "convergence_speed": 10,
        }
        for i, v in enumerate(delta_snrs)
    ]


def _make_sig_result(**overrides: Any) -> dict[str, Any]:
    """Build a significance result dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "signal_type": "ecg",
        "flicker_freq": 50.0,
        "input_snr": 10.0,
        "epochs": 50,
        "stdp_tau": 20.0,
        "n_seeds": 10,
        "mean_delta_snr": 5.0,
        "std_delta_snr": 0.5,
        "t_stat": 31.6,
        "p_value": 1e-10,
        "cohens_d": 10.0,
        "significant": True,
        "improved": True,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Step 3a: fetch_raw_runs
# ---------------------------------------------------------------------------


class TestFetchRawRuns:
    """Tests for fetch_raw_runs()."""

    def test_returns_all_rows(self, tmp_path: Path) -> None:
        """All inserted rows are returned."""
        db_path = tmp_path / "raw.db"
        _create_test_db(db_path, _sample_rows())
        rows = fetch_raw_runs(db_path)
        assert len(rows) == 3

    def test_empty_database(self, tmp_path: Path) -> None:
        """Empty database returns empty list."""
        db_path = tmp_path / "empty.db"
        _create_test_db(db_path, [])
        rows = fetch_raw_runs(db_path)
        assert rows == []

    def test_row_keys(self, tmp_path: Path) -> None:
        """Each row carries every column, Phase 3 additions included."""
        db_path = tmp_path / "keys.db"
        _create_test_db(db_path, _sample_rows())
        rows = fetch_raw_runs(db_path)
        expected = {
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
            "qrs_f1",
            "qrs_sensitivity",
            "qrs_ppv",
            "qrs_jitter_ms",
        }
        assert set(rows[0].keys()) == expected

    def test_pre_phase3_schema_gets_defaults(self, tmp_path: Path) -> None:
        """
        An archived 8-column database still reads.

        `_create_test_db` builds the pre-Phase-3 schema deliberately, so this
        exercises the `PRAGMA table_info` fallback rather than the migration.
        """
        db_path = tmp_path / "legacy.db"
        _create_test_db(db_path, _sample_rows())
        rows = fetch_raw_runs(db_path)
        assert all(r["dataset"] == "synthetic" for r in rows)
        assert all(r["record_id"] is None for r in rows)
        assert all(r["rpeak_integrity"] is None for r in rows)

    def test_row_values_match(self, tmp_path: Path) -> None:
        """delta_snr values match what was inserted."""
        db_path = tmp_path / "vals.db"
        _create_test_db(db_path, _sample_rows())
        rows = fetch_raw_runs(db_path)
        delta_snrs = {r["delta_snr"] for r in rows}
        assert delta_snrs == {4.0, 6.0, 8.0}


# ---------------------------------------------------------------------------
# Step 3b: compute_statistical_significance
# ---------------------------------------------------------------------------


class TestComputeStatisticalSignificance:
    """Tests for compute_statistical_significance()."""

    def test_clearly_significant(self) -> None:
        """Large consistent positive ΔSNR is significant."""
        results = compute_statistical_significance(_make_raw_runs(_CLEARLY_POSITIVE))
        assert len(results) == 1
        r = results[0]
        assert r["significant"] is True
        assert r["improved"] is True
        assert r["p_value"] < 0.05
        assert r["mean_delta_snr"] > 0.0

    def test_not_significant(self) -> None:
        """High-variance near-zero values are not significant."""
        results = compute_statistical_significance(_make_raw_runs(_INSIGNIFICANT))
        assert len(results) == 1
        assert results[0]["significant"] is False

    def test_negative_improvement(self) -> None:
        """Negative mean ΔSNR sets improved=False and significant=False."""
        results = compute_statistical_significance(_make_raw_runs(_CLEARLY_NEGATIVE))
        assert len(results) == 1
        r = results[0]
        assert r["improved"] is False
        assert r["significant"] is False

    def test_empty_input(self) -> None:
        """Empty run list returns empty result list."""
        assert compute_statistical_significance([]) == []

    def test_single_seed_no_ttest(self) -> None:
        """n=1 produces NaN p-value and significant=False (no t-test possible)."""
        results = compute_statistical_significance(_make_raw_runs([5.0]))
        assert len(results) == 1
        r = results[0]
        assert math.isnan(r["p_value"])
        assert r["significant"] is False

    def test_groups_by_config(self) -> None:
        """Different signal types produce separate result entries."""
        raw = _make_raw_runs(_CLEARLY_POSITIVE[:5], signal_type="ecg") + _make_raw_runs(
            _CLEARLY_POSITIVE[:5], signal_type="eeg"
        )
        results = compute_statistical_significance(raw)
        assert len(results) == 2
        assert {r["signal_type"] for r in results} == {"ecg", "eeg"}

    def test_result_keys(self) -> None:
        """Each result dict has the expected keys."""
        results = compute_statistical_significance(_make_raw_runs(_CLEARLY_POSITIVE))
        expected = {
            "dataset",
            "record_id",
            "signal_type",
            "flicker_freq",
            "input_snr",
            "epochs",
            "stdp_tau",
            "n_seeds",
            "mean_delta_snr",
            "std_delta_snr",
            "t_stat",
            "p_value",
            "cohens_d",
            "significant",
            "improved",
        }
        assert set(results[0].keys()) == expected

    def test_custom_p_threshold(self) -> None:
        """Looser threshold marks borderline configs as significant."""
        raw = _make_raw_runs(_INSIGNIFICANT)
        # p≈0.70: not significant at 0.05, significant at 0.90 (mean>0)
        strict = compute_statistical_significance(raw, p_threshold=0.05)
        loose = compute_statistical_significance(raw, p_threshold=0.90)
        assert strict[0]["significant"] is False
        assert loose[0]["significant"] is True

    def test_n_seeds_field(self) -> None:
        """n_seeds matches the number of input runs."""
        results = compute_statistical_significance(_make_raw_runs(_CLEARLY_POSITIVE))
        assert results[0]["n_seeds"] == len(_CLEARLY_POSITIVE)

    def test_cohens_d_positive_for_positive_mean(self) -> None:
        """Cohen's d is positive when mean ΔSNR is positive."""
        results = compute_statistical_significance(_make_raw_runs(_CLEARLY_POSITIVE))
        assert results[0]["cohens_d"] > 0.0


# ---------------------------------------------------------------------------
# Step 3c: summarize_significance
# ---------------------------------------------------------------------------


class TestSummarizeSignificance:
    """Tests for summarize_significance()."""

    def test_empty_results(self) -> None:
        """Empty input returns zeroed summary with None config pointers."""
        summary = summarize_significance([])
        assert summary["total_configs"] == 0
        assert summary["significant_configs"] == 0
        assert summary["best_config"] is None
        assert summary["worst_config"] is None

    def test_all_significant(self) -> None:
        """All significant configs → 100% pct_significant."""
        results = [_make_sig_result() for _ in range(3)]
        summary = summarize_significance(results)
        assert summary["total_configs"] == 3
        assert summary["significant_configs"] == 3
        assert summary["pct_significant"] == 100.0

    def test_none_significant(self) -> None:
        """No significant configs → 0% counts."""
        results = [
            _make_sig_result(significant=False, improved=False, mean_delta_snr=-1.0)
            for _ in range(3)
        ]
        summary = summarize_significance(results)
        assert summary["significant_configs"] == 0
        assert summary["improved_configs"] == 0
        assert summary["pct_significant"] == 0.0
        assert summary["pct_improved"] == 0.0

    def test_mixed_significance(self) -> None:
        """Mixed results produce correct counts and overall mean."""
        results = [
            _make_sig_result(mean_delta_snr=5.0, significant=True, improved=True),
            _make_sig_result(mean_delta_snr=0.5, significant=False, improved=True),
            _make_sig_result(mean_delta_snr=-1.0, significant=False, improved=False),
        ]
        summary = summarize_significance(results)
        assert summary["total_configs"] == 3
        assert summary["significant_configs"] == 1
        assert summary["improved_configs"] == 2
        expected_mean = (5.0 + 0.5 - 1.0) / 3
        assert abs(summary["mean_delta_snr_overall"] - expected_mean) < 1e-10

    def test_best_worst_config(self) -> None:
        """Best and worst configs are correctly identified."""
        results = [
            _make_sig_result(mean_delta_snr=5.0),
            _make_sig_result(mean_delta_snr=10.0),
            _make_sig_result(mean_delta_snr=-2.0, significant=False, improved=False),
        ]
        summary = summarize_significance(results)
        assert summary["best_config"]["mean_delta_snr"] == 10.0
        assert summary["worst_config"]["mean_delta_snr"] == -2.0


# ---------------------------------------------------------------------------
# Step 3d: identify_failure_modes
# ---------------------------------------------------------------------------


class TestIdentifyFailureModes:
    """Tests for identify_failure_modes()."""

    def test_no_failures(self) -> None:
        """All-passing configs produce an empty failure list."""
        results = [_make_sig_result() for _ in range(3)]
        assert identify_failure_modes(results) == []

    def test_negative_improvement(self) -> None:
        """Negative mean ΔSNR is flagged as negative_improvement."""
        results = [
            _make_sig_result(
                mean_delta_snr=-3.0,
                improved=False,
                significant=False,
                std_delta_snr=0.5,
            )
        ]
        failures = identify_failure_modes(results)
        assert len(failures) == 1
        assert "negative_improvement" in failures[0]["failure_reasons"]

    def test_not_significant(self) -> None:
        """Positive-but-insignificant improvement is flagged as not_significant."""
        results = [
            _make_sig_result(
                mean_delta_snr=0.5,
                significant=False,
                std_delta_snr=0.2,
                improved=True,
            )
        ]
        failures = identify_failure_modes(results)
        assert len(failures) == 1
        assert "not_significant" in failures[0]["failure_reasons"]

    def test_high_variance(self) -> None:
        """std > |mean| is flagged as high_variance even when significant."""
        results = [
            _make_sig_result(
                mean_delta_snr=1.0,
                std_delta_snr=3.0,
                significant=True,
                improved=True,
            )
        ]
        failures = identify_failure_modes(results)
        assert len(failures) == 1
        assert "high_variance" in failures[0]["failure_reasons"]

    def test_multiple_reasons(self) -> None:
        """Negative mean + large std produces both failure reasons."""
        results = [
            _make_sig_result(
                mean_delta_snr=-1.0,
                std_delta_snr=3.0,
                significant=False,
                improved=False,
            )
        ]
        failures = identify_failure_modes(results)
        assert len(failures) == 1
        reasons = failures[0]["failure_reasons"]
        assert "negative_improvement" in reasons
        assert "high_variance" in reasons

    def test_result_keys(self) -> None:
        """Failure dicts contain the expected keys."""
        results = [
            _make_sig_result(mean_delta_snr=-1.0, significant=False, improved=False)
        ]
        failures = identify_failure_modes(results)
        expected = {
            "signal_type",
            "flicker_freq",
            "input_snr",
            "epochs",
            "stdp_tau",
            "mean_delta_snr",
            "std_delta_snr",
            "p_value",
            "failure_reasons",
        }
        assert set(failures[0].keys()) == expected

    def test_mixed_configs(self) -> None:
        """Only failing configs are returned from a mixed list."""
        results = [
            _make_sig_result(mean_delta_snr=5.0),
            _make_sig_result(
                mean_delta_snr=-2.0,
                improved=False,
                significant=False,
                std_delta_snr=0.5,
            ),
            _make_sig_result(
                mean_delta_snr=0.3,
                significant=False,
                std_delta_snr=0.1,
                improved=True,
            ),
        ]
        failures = identify_failure_modes(results)
        assert len(failures) == 2
