"""Comprehensive tests for CLI, runner, and database (Phase 2).

T11 — 4 steps:
  Step 1: Phase2Config validation
  Step 2: ResultsDB
  Step 3: run_single_experiment
  Step 4: Integration test for run_experiment_command
"""

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml
from neuroflux_datasets.cli import (
    Phase2Config,
    ResultsDB,
    RunResult,
    run_experiment_command,
    run_single_experiment,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Step 1: Phase2Config validation
# ---------------------------------------------------------------------------


class TestPhase2Config:
    """Tests for Pydantic Phase2Config schema."""

    def test_valid_config(self) -> None:
        """A complete, valid config parses without error."""
        cfg = Phase2Config(
            signal_types=["ecg", "eeg", "sine"],
            flicker_frequencies=[50.0, 60.0],
            input_snrs=[0.0, 10.0],
            training_epochs=[10, 50],
            stdp_taus=[20.0],
            seeds_per_config=5,
            duration_s=2.0,
        )
        assert cfg.signal_types == ["ecg", "eeg", "sine"]
        assert cfg.flicker_frequencies == [50.0, 60.0]
        assert cfg.seeds_per_config == 5
        assert cfg.duration_s == 2.0

    def test_defaults(self) -> None:
        """seeds_per_config and duration_s have correct defaults."""
        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[50],
            stdp_taus=[20.0],
        )
        assert cfg.seeds_per_config == 10
        assert cfg.duration_s == 1.0

    def test_invalid_signal_type(self) -> None:
        """Invalid signal type is rejected by Literal constraint."""
        with pytest.raises(ValidationError):
            Phase2Config(
                signal_types=["ecg", "ppg"],  # type: ignore[list-item]
                flicker_frequencies=[50.0],
                input_snrs=[10.0],
                training_epochs=[50],
                stdp_taus=[20.0],
            )

    def test_missing_required_field(self) -> None:
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            Phase2Config(
                signal_types=["ecg"],
                # missing flicker_frequencies, input_snrs, etc.
            )  # type: ignore[call-arg]

    def test_wrong_type_for_field(self) -> None:
        """String where float list expected raises ValidationError."""
        with pytest.raises(ValidationError):
            Phase2Config(
                signal_types=["ecg"],
                flicker_frequencies="not a list",  # type: ignore[arg-type]
                input_snrs=[10.0],
                training_epochs=[50],
                stdp_taus=[20.0],
            )

    def test_empty_signal_types(self) -> None:
        """Empty signal_types list is technically valid Pydantic."""
        cfg = Phase2Config(
            signal_types=[],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[50],
            stdp_taus=[20.0],
        )
        assert cfg.signal_types == []

    def test_flicker_amplitude_default(self) -> None:
        """flicker_amplitude defaults to 0.5."""
        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[50],
            stdp_taus=[20.0],
        )
        assert cfg.flicker_amplitude == 0.5

    def test_flicker_amplitude_custom(self) -> None:
        """flicker_amplitude accepts non-default values."""
        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[50],
            stdp_taus=[20.0],
            flicker_amplitude=1.0,
        )
        assert cfg.flicker_amplitude == 1.0

    def test_from_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Config can be serialized to YAML and parsed back."""
        original = Phase2Config(
            signal_types=["ecg", "sine"],
            flicker_frequencies=[50.0, 100.0],
            input_snrs=[5.0, 10.0],
            training_epochs=[50, 100],
            stdp_taus=[10.0, 20.0],
            seeds_per_config=3,
            duration_s=0.5,
        )
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(original.model_dump()))

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        restored = Phase2Config(**raw)

        assert restored == original


# ---------------------------------------------------------------------------
# Step 2: ResultsDB
# ---------------------------------------------------------------------------


def _make_result(**overrides: Any) -> RunResult:
    """Create a RunResult with sensible defaults, overridable."""
    defaults: dict[str, Any] = {
        "signal_type": "ecg",
        "flicker_freq": 50.0,
        "input_snr": 10.0,
        "epochs": 50,
        "stdp_tau": 20.0,
        "seed": 0,
        "delta_snr": 5.0,
        "convergence_speed": 15,
    }
    defaults.update(overrides)
    return RunResult(**defaults)


class TestResultsDB:
    """Tests for the SQLite ResultsDB backend."""

    def test_db_file_created(self, tmp_path: Path) -> None:
        """Database file is created at the specified path."""
        db_path = tmp_path / "results.db"
        ResultsDB(db_path)
        assert db_path.exists()

    def test_db_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        db_path = tmp_path / "deep" / "nested" / "dir" / "results.db"
        ResultsDB(db_path)
        assert db_path.exists()

    def test_schema_correct(self, tmp_path: Path) -> None:
        """The 'runs' table exists with expected columns."""
        db_path = tmp_path / "schema.db"
        ResultsDB(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(runs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id",
            "signal_type",
            "flicker_freq",
            "input_snr",
            "epochs",
            "stdp_tau",
            "seed",
            "delta_snr",
            "convergence_speed",
        }
        assert columns == expected

    def test_insert_and_query(self, tmp_path: Path) -> None:
        """Inserted result can be queried back correctly."""
        db_path = tmp_path / "insert.db"
        db = ResultsDB(db_path)

        res = _make_result(delta_snr=7.5, convergence_speed=12)
        db.insert_result(res)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM runs").fetchall()
        conn.close()

        assert len(rows) == 1
        # Columns: id, signal_type, flicker_freq, input_snr, epochs,
        #          stdp_tau, seed, delta_snr, convergence_speed
        row = rows[0]
        assert row[1] == "ecg"
        assert row[2] == 50.0
        assert row[3] == 10.0
        assert row[4] == 50
        assert row[5] == 20.0
        assert row[6] == 0
        assert abs(row[7] - 7.5) < 1e-10
        assert row[8] == 12

    def test_multiple_inserts(self, tmp_path: Path) -> None:
        """Multiple inserts accumulate rows."""
        db_path = tmp_path / "multi.db"
        db = ResultsDB(db_path)

        for i in range(5):
            db.insert_result(_make_result(seed=i, delta_snr=float(i)))

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 5

    def test_duplicate_data_allowed(self, tmp_path: Path) -> None:
        """Inserting identical data twice creates two rows (no uniqueness)."""
        db_path = tmp_path / "dup.db"
        db = ResultsDB(db_path)

        res = _make_result()
        db.insert_result(res)
        db.insert_result(res)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 2

    def test_negative_delta_snr(self, tmp_path: Path) -> None:
        """Negative delta_snr values are stored correctly."""
        db_path = tmp_path / "neg.db"
        db = ResultsDB(db_path)

        db.insert_result(_make_result(delta_snr=-3.2))

        conn = sqlite3.connect(str(db_path))
        val = conn.execute("SELECT delta_snr FROM runs").fetchone()[0]
        conn.close()
        assert abs(val - (-3.2)) < 1e-10


# ---------------------------------------------------------------------------
# Step 3: run_single_experiment
# ---------------------------------------------------------------------------


def _minimal_params(**overrides: Any) -> dict[str, Any]:
    """Minimal valid parameter dict for run_single_experiment."""
    defaults: dict[str, Any] = {
        "signal_type": "ecg",
        "flicker_freq": 50.0,
        "input_snr": 10.0,
        "epochs": 5,
        "stdp_tau": 20.0,
        "seed": 42,
        "duration_s": 0.5,
    }
    defaults.update(overrides)
    return defaults


@pytest.mark.slow
class TestRunSingleExperiment:
    """Tests for run_single_experiment().

    Uses small epoch counts and short durations for speed.
    """

    def test_returns_run_result(self) -> None:
        """Returns a RunResult with all fields populated."""
        result = run_single_experiment(_minimal_params())
        assert isinstance(result, RunResult)
        assert result.signal_type == "ecg"
        assert result.flicker_freq == 50.0
        assert result.input_snr == 10.0
        assert result.epochs == 5
        assert result.stdp_tau == 20.0
        assert result.seed == 42

    def test_delta_snr_is_finite(self) -> None:
        """delta_snr must be a finite float (not NaN, not inf)."""
        result = run_single_experiment(_minimal_params())
        assert math.isfinite(result.delta_snr)

    def test_convergence_speed_valid(self) -> None:
        """convergence_speed is a positive int ≤ epochs."""
        params = _minimal_params(epochs=10)
        result = run_single_experiment(params)
        assert isinstance(result.convergence_speed, int)
        assert 0 < result.convergence_speed <= 10

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed → identical delta_snr."""
        r1 = run_single_experiment(_minimal_params(seed=99))
        r2 = run_single_experiment(_minimal_params(seed=99))
        assert r1.delta_snr == r2.delta_snr
        assert r1.convergence_speed == r2.convergence_speed

    def test_different_seeds_diverge(self) -> None:
        """Different seeds → different delta_snr values."""
        r1 = run_single_experiment(_minimal_params(seed=0))
        r2 = run_single_experiment(_minimal_params(seed=1))
        assert r1.delta_snr != r2.delta_snr

    def test_ecg_signal_type(self) -> None:
        """ECG signal type runs without error."""
        result = run_single_experiment(_minimal_params(signal_type="ecg"))
        assert result.signal_type == "ecg"
        assert math.isfinite(result.delta_snr)

    def test_eeg_signal_type(self) -> None:
        """EEG signal type runs without error."""
        result = run_single_experiment(_minimal_params(signal_type="eeg"))
        assert result.signal_type == "eeg"
        assert math.isfinite(result.delta_snr)

    def test_sine_signal_type(self) -> None:
        """Sine signal type runs without error."""
        result = run_single_experiment(_minimal_params(signal_type="sine"))
        assert result.signal_type == "sine"
        assert math.isfinite(result.delta_snr)

    def test_single_epoch(self) -> None:
        """epochs=1 still produces a valid result."""
        result = run_single_experiment(_minimal_params(epochs=1))
        assert math.isfinite(result.delta_snr)
        assert result.convergence_speed >= 1

    def test_custom_flicker_amplitude(self) -> None:
        """Non-default flicker_amplitude runs without error."""
        result = run_single_experiment(_minimal_params(flicker_amplitude=1.0))
        assert math.isfinite(result.delta_snr)

    def test_flicker_amplitude_affects_result(self) -> None:
        """Different flicker_amplitude values produce different delta_snr."""
        r_low = run_single_experiment(_minimal_params(seed=7, flicker_amplitude=0.1))
        r_high = run_single_experiment(_minimal_params(seed=7, flicker_amplitude=1.0))
        assert r_low.delta_snr != r_high.delta_snr


# ---------------------------------------------------------------------------
# Step 4: Integration test for run_experiment_command
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunExperimentCommand:
    """Integration test: config YAML → run_experiment_command → DB."""

    def test_end_to_end(self, tmp_path: Path) -> None:
        """Full pipeline: write config, run command, verify DB rows."""
        # Create a minimal config: 1 signal × 1 freq × 1 snr × 1 epoch
        #                          × 1 tau × 1 seed = 1 run
        config = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 1,
            "duration_s": 0.5,
        }
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(config))

        db_path = tmp_path / "test_results.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        # Verify DB exists and has exactly 1 row
        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM runs").fetchall()
        conn.close()

        assert len(rows) == 1
        # Verify the row has the correct config values
        row = rows[0]
        assert row[1] == "ecg"  # signal_type
        assert row[2] == 50.0  # flicker_freq
        assert row[3] == 10.0  # input_snr
        assert row[4] == 5  # epochs
        assert row[5] == 20.0  # stdp_tau
        assert row[6] == 0  # seed (0-indexed)
        assert math.isfinite(row[7])  # delta_snr

    def test_multi_seed_config(self, tmp_path: Path) -> None:
        """Multiple seeds produce the expected number of rows."""
        config = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 3,
            "duration_s": 0.5,
        }
        config_path = tmp_path / "multi_seed.yaml"
        config_path.write_text(yaml.dump(config))

        db_path = tmp_path / "multi_seed.db"
        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 3

    def test_multi_param_config(self, tmp_path: Path) -> None:
        """Cartesian product of params produces correct row count."""
        config = {
            "signal_types": ["ecg", "sine"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 2,
            "duration_s": 0.5,
        }
        # Expected: 2 signals × 1 × 1 × 1 × 1 × 2 seeds = 4 runs
        config_path = tmp_path / "multi_param.yaml"
        config_path.write_text(yaml.dump(config))

        db_path = tmp_path / "multi_param.db"
        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        signal_types = conn.execute("SELECT DISTINCT signal_type FROM runs").fetchall()
        conn.close()

        assert count == 4
        assert len(signal_types) == 2

    def test_flicker_amplitude_in_config(self, tmp_path: Path) -> None:
        """Config with explicit flicker_amplitude runs and produces rows."""
        config = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 1,
            "duration_s": 0.5,
            "flicker_amplitude": 1.0,
        }
        config_path = tmp_path / "flicker_amp.yaml"
        config_path.write_text(yaml.dump(config))
        db_path = tmp_path / "flicker_amp.db"
        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 1

    def test_results_are_queryable(self, tmp_path: Path) -> None:
        """Results written by runner can be aggregated by analysis."""
        from neuroflux_datasets.analysis import fetch_aggregated_data

        config = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 2,
            "duration_s": 0.5,
        }
        config_path = tmp_path / "queryable.yaml"
        config_path.write_text(yaml.dump(config))

        db_path = tmp_path / "queryable.db"
        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        # Verify the analysis module can read and aggregate
        results = fetch_aggregated_data(db_path)
        assert len(results) == 1  # 1 config group, 2 seeds averaged
        assert results[0]["signal_type"] == "ecg"
        assert math.isfinite(results[0]["mean_delta_snr"])
