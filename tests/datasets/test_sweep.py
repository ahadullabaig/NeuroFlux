"""Full sweep execution and end-to-end pipeline validation (Phase 2).

T12 — 3 steps:
  Step 1: Validate configs load correctly
  Step 2: Execute quick validation sweep
  Step 3: Verify analysis pipeline end-to-end
"""

import argparse
import itertools
import math
import sqlite3
from pathlib import Path

import pytest
import yaml
from neuroflux_datasets.cli import (
    Phase2Config,
    run_experiment_command,
)

# Root of the project — configs/ is relative to this
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Step 1: Validate configs load correctly
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Verify both YAML config files parse and produce correct run counts."""

    def test_full_sweep_config_loads(self) -> None:
        """full_sweep.yaml parses and produces 5,760 runs."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "full_sweep.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        cfg = Phase2Config(**raw)

        # 3 signals × 4 freqs × 4 SNRs × 4 epochs × 3 taus = 576 combos
        combos = len(
            list(
                itertools.product(
                    cfg.signal_types,
                    cfg.flicker_frequencies,
                    cfg.input_snrs,
                    cfg.training_epochs,
                    cfg.stdp_taus,
                )
            )
        )
        assert combos == 576

        # × 10 seeds = 5,760 total runs
        total_runs = combos * cfg.seeds_per_config
        assert total_runs == 5760

    def test_full_sweep_config_values(self) -> None:
        """full_sweep.yaml matches PLAN.md specifications."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "full_sweep.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        cfg = Phase2Config(**raw)
        assert cfg.signal_types == ["ecg", "eeg", "sine"]
        assert cfg.flicker_frequencies == [50.0, 60.0, 100.0, 120.0]
        assert cfg.input_snrs == [0.0, 5.0, 10.0, 20.0]
        assert cfg.training_epochs == [10, 50, 100, 500]
        assert cfg.stdp_taus == [10.0, 20.0, 40.0]
        assert cfg.seeds_per_config == 10
        assert cfg.duration_s == 1.0

    def test_quick_validation_config_loads(self) -> None:
        """quick_validation.yaml parses and produces 2 runs."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        cfg = Phase2Config(**raw)

        combos = len(
            list(
                itertools.product(
                    cfg.signal_types,
                    cfg.flicker_frequencies,
                    cfg.input_snrs,
                    cfg.training_epochs,
                    cfg.stdp_taus,
                )
            )
        )
        total_runs = combos * cfg.seeds_per_config
        assert total_runs == 2

    def test_quick_validation_config_values(self) -> None:
        """quick_validation.yaml has minimal parameters."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        cfg = Phase2Config(**raw)
        assert cfg.signal_types == ["ecg"]
        assert cfg.flicker_frequencies == [50.0]
        assert cfg.input_snrs == [10.0]
        assert cfg.training_epochs == [50]
        assert cfg.stdp_taus == [20.0]
        assert cfg.seeds_per_config == 2
        assert cfg.duration_s == 1.0


# ---------------------------------------------------------------------------
# Step 2: Execute quick validation sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQuickValidationSweep:
    """Run the quick_validation config and verify DB results."""

    def test_sweep_produces_correct_row_count(
        self,
        tmp_path: Path,
    ) -> None:
        """Quick validation sweep produces exactly 2 rows."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        db_path = tmp_path / "quick_results.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert count == 2

    def test_sweep_delta_snr_reasonable(
        self,
        tmp_path: Path,
    ) -> None:
        """All delta_snr values are finite and not extreme."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        db_path = tmp_path / "reasonable.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT delta_snr, convergence_speed FROM runs").fetchall()
        conn.close()

        for delta_snr, conv_speed in rows:
            assert math.isfinite(delta_snr), f"delta_snr is not finite: {delta_snr}"
            # delta_snr should be in a physically plausible range
            assert -50.0 < delta_snr < 50.0, f"delta_snr out of range: {delta_snr}"
            assert isinstance(conv_speed, int)
            assert conv_speed > 0

    def test_sweep_seeds_differ(
        self,
        tmp_path: Path,
    ) -> None:
        """Different seeds produce different delta_snr values."""
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        db_path = tmp_path / "seeds_differ.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        conn = sqlite3.connect(str(db_path))
        snrs = conn.execute("SELECT delta_snr FROM runs ORDER BY seed").fetchall()
        conn.close()

        assert len(snrs) == 2
        # Two different seeds should produce different SNR values
        assert snrs[0][0] != snrs[1][0]


# ---------------------------------------------------------------------------
# Step 3: Verify analysis pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAnalysisPipeline:
    """Run analysis on sweep results and verify plot generation."""

    def test_analysis_generates_plots(
        self,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: config → run → analyze → plot files."""
        from neuroflux_datasets.analysis import (
            fetch_aggregated_data,
            plot_convergence_curves,
            plot_parameter_sensitivity,
            plot_snr_heatmap,
        )

        # Step A: Run sweep
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        db_path = tmp_path / "analysis_test.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        # Step B: Fetch aggregated data
        results = fetch_aggregated_data(db_path)
        assert len(results) >= 1
        assert results[0]["signal_type"] == "ecg"
        assert math.isfinite(results[0]["mean_delta_snr"])

        # Step C: Generate all three plot types
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        heatmap_path = figures_dir / "snr_heatmap_ecg.png"
        plot_snr_heatmap(results, "ecg", heatmap_path)
        assert heatmap_path.exists()
        assert heatmap_path.stat().st_size > 0

        convergence_path = figures_dir / "convergence_ecg.png"
        plot_convergence_curves(results, "ecg", convergence_path)
        assert convergence_path.exists()
        assert convergence_path.stat().st_size > 0

        sensitivity_path = figures_dir / "sensitivity_ecg.png"
        plot_parameter_sensitivity(results, "ecg", sensitivity_path)
        assert sensitivity_path.exists()
        assert sensitivity_path.stat().st_size > 0

    def test_analysis_aggregation_correct(
        self,
        tmp_path: Path,
    ) -> None:
        """Aggregated results average over seeds correctly."""
        from neuroflux_datasets.analysis import fetch_aggregated_data

        # Run sweep (2 seeds, 1 config)
        config_path = PROJECT_ROOT / "configs" / "phase2" / "quick_validation.yaml"
        db_path = tmp_path / "agg_test.db"

        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
        )
        run_experiment_command(args)

        # Get raw values
        conn = sqlite3.connect(str(db_path))
        raw_snrs = conn.execute("SELECT delta_snr FROM runs").fetchall()
        conn.close()
        expected_mean = sum(r[0] for r in raw_snrs) / len(raw_snrs)

        # Get aggregated values
        results = fetch_aggregated_data(db_path)
        assert len(results) == 1  # 1 config group
        actual_mean = results[0]["mean_delta_snr"]

        assert abs(actual_mean - expected_mean) < 1e-10
