"""Additional coverage tests for ``neuroflux_datasets.cli``.

Complements ``test_cli.py``, ``test_cli_phase3.py``, ``test_sweep.py``,
``test_results_db_migration.py`` by exercising:

- ``extra="forbid"`` on both Pydantic configs
- ``_compute_convergence_speed`` edge cases
- ``run_single_experiment`` with a constant signal
- The Phase 2 dry-run branch
- ``analyze_significance_command`` — empty DB, OperationalError, success
- ``analyze_experiment_command`` — empty DB, OperationalError, success
- ``main()`` argparse dispatch for all three subcommands
- ``run_experiment_command`` raising RuntimeError on per-task failures
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from neuroflux_datasets import cli as cli_mod
from neuroflux_datasets.cli import (
    Phase2Config,
    Phase3Config,
    ResultsDB,
    RunResult,
    _compute_convergence_speed,
    analyze_experiment_command,
    analyze_significance_command,
    main,
    run_experiment_command,
    run_single_experiment,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# extra="forbid" — typo'd YAML keys must fail loudly
# ---------------------------------------------------------------------------


class TestExtraForbid:
    """Both Phase2Config and Phase3Config reject unknown fields."""

    def test_phase2_rejects_unknown_field(self) -> None:
        """A typo'd ``signal_type`` (vs ``signal_types``) is a ValidationError."""
        # Dict splat dodges mypy's call-arg checking for the deliberate typo.
        bad_kwargs: dict[str, Any] = {
            "signal_type": ["ecg"],  # typo
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
        }
        with pytest.raises(ValidationError):
            Phase2Config(**bad_kwargs)

    def test_phase3_rejects_unknown_field(self) -> None:
        """An unknown key in Phase3Config raises ValidationError."""
        bad_kwargs: dict[str, Any] = {
            "record_ids": ["100"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "bogus_field": 42,
        }
        with pytest.raises(ValidationError):
            Phase3Config(**bad_kwargs)


# ---------------------------------------------------------------------------
# network_layers config field
# ---------------------------------------------------------------------------


class TestNetworkLayersField:
    """Both Phase2Config and Phase3Config carry a configurable network shape."""

    def test_phase2_default_is_32_16_8(self) -> None:
        """Omitting ``network_layers`` keeps the pre-Phase-3 default."""
        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
        )
        assert cfg.network_layers == [32, 16, 8]

    def test_phase3_default_is_32_16_8(self) -> None:
        cfg = Phase3Config(
            record_ids=["100"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
        )
        assert cfg.network_layers == [32, 16, 8]

    def test_phase2_accepts_explicit_layers(self) -> None:
        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
            network_layers=[16, 8, 4],
        )
        assert cfg.network_layers == [16, 8, 4]

    def test_phase3_accepts_explicit_layers(self) -> None:
        cfg = Phase3Config(
            record_ids=["100"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
            network_layers=[64, 32, 64],
        )
        assert cfg.network_layers == [64, 32, 64]

    def test_phase2_rejects_network_layer_typo(self) -> None:
        """The singular ``network_layer`` typo must hit extra=forbid."""
        bad: dict[str, Any] = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [10],
            "stdp_taus": [20.0],
            "network_layer": [16, 8, 4],  # typo
        }
        with pytest.raises(ValidationError):
            Phase2Config(**bad)

    def test_phase3_rejects_network_layer_typo(self) -> None:
        bad: dict[str, Any] = {
            "record_ids": ["100"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [10],
            "stdp_taus": [20.0],
            "network_layer": [16, 8, 4],  # typo
        }
        with pytest.raises(ValidationError):
            Phase3Config(**bad)

    def test_build_phase2_tasks_carries_network_layers(self) -> None:
        from neuroflux_datasets.cli import _build_phase2_tasks

        cfg = Phase2Config(
            signal_types=["ecg"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
            network_layers=[16, 8, 4],
            seeds_per_config=1,
        )
        tasks = _build_phase2_tasks(cfg)
        assert all(t["network_layers"] == [16, 8, 4] for t in tasks)

    def test_build_phase3_tasks_carries_network_layers(self) -> None:
        from neuroflux_datasets.cli import _build_phase3_tasks

        cfg = Phase3Config(
            record_ids=["100"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[10],
            stdp_taus=[20.0],
            network_layers=[64, 32, 64],
            seeds_per_config=1,
        )
        tasks = _build_phase3_tasks(cfg)
        assert all(t["network_layers"] == [64, 32, 64] for t in tasks)

    def test_run_single_experiment_respects_network_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing a smaller network_layers actually builds a smaller network."""
        n = 250
        clean_sig = np.sin(2 * np.pi * np.arange(n) / n).astype(np.float64)
        noisy_sig = clean_sig + np.random.default_rng(0).normal(0, 0.3, n)

        def _fake_pair(
            params: dict[str, Any], fs: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return clean_sig, noisy_sig

        monkeypatch.setattr(cli_mod, "_resolve_signal", _fake_pair)

        from neuroflux_core.network.feedforward import Network as _Network

        captured: dict[str, Any] = {}
        original_feedforward = _Network.feedforward

        def _capture(*args: Any, **kwargs: Any) -> Any:
            captured["layers"] = kwargs.get("layers")
            return original_feedforward(*args, **kwargs)

        monkeypatch.setattr(_Network, "feedforward", staticmethod(_capture))

        params = {
            "signal_type": "ecg",
            "flicker_freq": 50.0,
            "input_snr": 10.0,
            "epochs": 1,
            "stdp_tau": 20.0,
            "seed": 0,
            "duration_s": 1.0,
            "network_layers": [16, 8, 4],
        }
        res = run_single_experiment(params)

        assert captured["layers"] == [16, 8, 4]
        assert res.epochs == 1

    def test_run_single_experiment_defaults_when_layers_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task dicts without network_layers fall back to [32, 16, 8]."""
        n = 250
        clean_sig = np.sin(2 * np.pi * np.arange(n) / n).astype(np.float64)
        noisy_sig = clean_sig + np.random.default_rng(0).normal(0, 0.3, n)

        def _fake_pair(
            params: dict[str, Any], fs: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return clean_sig, noisy_sig

        monkeypatch.setattr(cli_mod, "_resolve_signal", _fake_pair)

        from neuroflux_core.network.feedforward import Network as _Network

        captured: dict[str, Any] = {}
        original_feedforward = _Network.feedforward

        def _capture(*args: Any, **kwargs: Any) -> Any:
            captured["layers"] = kwargs.get("layers")
            return original_feedforward(*args, **kwargs)

        monkeypatch.setattr(_Network, "feedforward", staticmethod(_capture))

        params = {
            "signal_type": "ecg",
            "flicker_freq": 50.0,
            "input_snr": 10.0,
            "epochs": 1,
            "stdp_tau": 20.0,
            "seed": 0,
            "duration_s": 1.0,
        }
        run_single_experiment(params)

        assert captured["layers"] == [32, 16, 8]


# ---------------------------------------------------------------------------
# _compute_convergence_speed — the line-188 fallthrough
# ---------------------------------------------------------------------------


class TestComputeConvergenceSpeed:
    """Edge cases for the convergence-speed helper."""

    def test_threshold_never_reached_returns_total_epochs(self) -> None:
        """final_delta_snr > 0 but no epoch reaches 90% → returns len(epochs)."""
        epoch_snrs = [0.1, 0.2, 0.3]
        final = 10.0  # 0.9 * 10.0 = 9.0; never reached
        assert _compute_convergence_speed(epoch_snrs, final) == 3

    def test_negative_final_snr_returns_total_epochs(self) -> None:
        """final_delta_snr ≤ 0 short-circuits to len(epochs)."""
        assert _compute_convergence_speed([1.0, 2.0, 3.0], -1.0) == 3

    def test_empty_epochs_returns_zero(self) -> None:
        """Empty epoch list with non-positive final returns 0."""
        assert _compute_convergence_speed([], 0.0) == 0

    def test_threshold_reached_at_first_epoch(self) -> None:
        """If the first epoch already exceeds 0.9*final, return 1."""
        assert _compute_convergence_speed([10.0, 11.0], 10.0) == 1


# ---------------------------------------------------------------------------
# run_single_experiment — v_max <= v_min fallback path
# ---------------------------------------------------------------------------


class TestConstantSignalFallback:
    """Cover the ``v_max = v_min + 1.0`` branch in ``run_single_experiment``."""

    def test_constant_noisy_signal_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When noisy is constant, the encoder gets v_min/v_max=v_min+1."""
        n = 250  # 1 s at 250 Hz
        const = np.ones(n, dtype=np.float64)

        def _const_pair(
            params: dict[str, Any], fs: float
        ) -> tuple[np.ndarray, np.ndarray]:
            return const.copy(), const.copy()

        monkeypatch.setattr(cli_mod, "_resolve_signal", _const_pair)

        params = {
            "signal_type": "ecg",
            "flicker_freq": 50.0,
            "input_snr": 10.0,
            "epochs": 1,
            "stdp_tau": 20.0,
            "seed": 0,
            "duration_s": 1.0,
        }
        res = run_single_experiment(params)
        # Constant clean=noisy: base_snr is ill-defined but the run must complete.
        assert res.dataset == "synthetic"
        assert res.epochs == 1


# ---------------------------------------------------------------------------
# Phase 2 dry-run — the synthetic branch of the dry-run printout
# ---------------------------------------------------------------------------


class TestPhase2DryRun:
    """Verify the Phase 2 dry-run branch (default ``dataset`` key)."""

    def test_dry_run_creates_no_database(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Phase 2 dry-run prints the synthetic config and opens no DB."""
        cfg_path = tmp_path / "phase2.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "signal_types": ["ecg", "sine"],
                    "flicker_frequencies": [50.0, 60.0],
                    "input_snrs": [10.0],
                    "training_epochs": [5],
                    "stdp_taus": [20.0],
                    "seeds_per_config": 2,
                    "duration_s": 1.0,
                }
            )
        )
        db_path = tmp_path / "must_not_exist.db"
        args = argparse.Namespace(
            config=str(cfg_path),
            output_db=str(db_path),
            dry_run=True,
        )

        run_experiment_command(args)

        out = capsys.readouterr().out
        assert "Dry run:" in out
        assert "synthetic" in out
        assert "Signal types" in out
        assert not db_path.exists()


# ---------------------------------------------------------------------------
# analyze-significance command — three branches
# ---------------------------------------------------------------------------


def _make_db_with_runs(db_path: Path, runs: list[RunResult]) -> None:
    """Helper: create a fresh DB and insert the given RunResult rows."""
    db = ResultsDB(db_path)
    for r in runs:
        db.insert_result(r)


class TestAnalyzeSignificanceCommand:
    """Exercise the three branches of analyze_significance_command."""

    def test_missing_table_prints_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An OperationalError on read is caught and printed."""
        # Create a DB file with no ``runs`` table to force OperationalError.
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()

        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(tmp_path / "sig"),
            p_threshold=0.05,
        )
        analyze_significance_command(args)
        out = capsys.readouterr().out
        assert "Error reading database" in out

    def test_empty_db_prints_no_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A DB with the runs table but zero rows prints 'No runs found'."""
        db_path = tmp_path / "empty.db"
        ResultsDB(db_path)  # creates empty schema

        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(tmp_path / "sig"),
            p_threshold=0.05,
        )
        analyze_significance_command(args)
        out = capsys.readouterr().out
        assert "No runs found" in out

    def test_success_writes_csvs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With real rows, CSV reports are written and summary printed."""
        db_path = tmp_path / "good.db"
        # Multiple seeds for the same config so a t-test has variance.
        runs = [
            RunResult(
                signal_type="ecg",
                flicker_freq=50.0,
                input_snr=10.0,
                epochs=5,
                stdp_tau=20.0,
                seed=s,
                delta_snr=4.0 + 0.1 * s,
                convergence_speed=2,
            )
            for s in range(5)
        ]
        _make_db_with_runs(db_path, runs)

        out_dir = tmp_path / "sig"
        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(out_dir),
            p_threshold=0.05,
        )
        analyze_significance_command(args)

        captured = capsys.readouterr().out
        assert "Significance Summary" in captured
        assert "Total configs" in captured
        assert (out_dir / "significance_results.csv").exists()


# ---------------------------------------------------------------------------
# analyze command — three branches
# ---------------------------------------------------------------------------


class TestAnalyzeExperimentCommand:
    """Exercise the three branches of analyze_experiment_command."""

    def test_missing_table_prints_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """OperationalError → printed, not raised."""
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()

        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(tmp_path / "figs"),
        )
        analyze_experiment_command(args)
        out = capsys.readouterr().out
        assert "Error reading database" in out

    def test_empty_db_prints_no_results(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty DB → 'No results found'."""
        db_path = tmp_path / "empty.db"
        ResultsDB(db_path)

        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(tmp_path / "figs"),
        )
        analyze_experiment_command(args)
        out = capsys.readouterr().out
        assert "No results found" in out

    def test_success_writes_plots(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With rows, each plot function is called once per signal_type."""
        called: dict[str, list[str]] = {
            "heatmap": [],
            "convergence": [],
            "sensitivity": [],
        }

        def _fake_plot(
            results: list[dict[str, Any]],
            signal_type: str,
            output_path: str | Path,
            *,
            kind: str,
        ) -> None:
            called[kind].append(signal_type)

        monkeypatch.setattr(
            "neuroflux_datasets.analysis.plot_snr_heatmap",
            lambda r, s, p: _fake_plot(r, s, p, kind="heatmap"),
        )
        monkeypatch.setattr(
            "neuroflux_datasets.analysis.plot_convergence_curves",
            lambda r, s, p: _fake_plot(r, s, p, kind="convergence"),
        )
        monkeypatch.setattr(
            "neuroflux_datasets.analysis.plot_parameter_sensitivity",
            lambda r, s, p: _fake_plot(r, s, p, kind="sensitivity"),
        )

        db_path = tmp_path / "good.db"
        runs = [
            RunResult(
                signal_type=st,
                flicker_freq=50.0,
                input_snr=10.0,
                epochs=5,
                stdp_tau=20.0,
                seed=0,
                delta_snr=3.0,
                convergence_speed=2,
            )
            for st in ("ecg", "sine")
        ]
        _make_db_with_runs(db_path, runs)

        args = argparse.Namespace(
            db=str(db_path),
            output_dir=str(tmp_path / "figs"),
        )
        analyze_experiment_command(args)
        capsys.readouterr()

        assert sorted(called["heatmap"]) == ["ecg", "sine"]
        assert sorted(called["convergence"]) == ["ecg", "sine"]
        assert sorted(called["sensitivity"]) == ["ecg", "sine"]


# ---------------------------------------------------------------------------
# main() — argparse dispatch for all three subcommands
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """Drive ``main()`` through every subcommand via sys.argv patching."""

    def test_run_subcommand_routes_to_run_experiment_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``neuroflux-experiment run cfg --dry-run`` calls run_experiment_command."""
        called: dict[str, argparse.Namespace] = {}

        def _capture(args: argparse.Namespace) -> None:
            called["args"] = args

        monkeypatch.setattr(cli_mod, "run_experiment_command", _capture)

        cfg = tmp_path / "c.yaml"
        cfg.write_text("signal_types: [ecg]\n")
        monkeypatch.setattr(
            sys,
            "argv",
            ["neuroflux-experiment", "run", str(cfg), "--dry-run"],
        )
        main()
        assert called["args"].config == str(cfg)
        assert called["args"].dry_run is True

    def test_analyze_subcommand_routes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``neuroflux-experiment analyze db`` calls analyze_experiment_command."""
        called: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(
            cli_mod,
            "analyze_experiment_command",
            lambda args: called.setdefault("args", args),
        )

        db = tmp_path / "x.db"
        db.touch()
        monkeypatch.setattr(
            sys,
            "argv",
            ["neuroflux-experiment", "analyze", str(db), "--output-dir", str(tmp_path)],
        )
        main()
        assert called["args"].db == str(db)
        assert called["args"].output_dir == str(tmp_path)

    def test_analyze_significance_subcommand_routes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``analyze-significance`` routes with the p-threshold argument."""
        called: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(
            cli_mod,
            "analyze_significance_command",
            lambda args: called.setdefault("args", args),
        )

        db = tmp_path / "x.db"
        db.touch()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "neuroflux-experiment",
                "analyze-significance",
                str(db),
                "--p-threshold",
                "0.01",
            ],
        )
        main()
        assert called["args"].db == str(db)
        assert called["args"].p_threshold == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# run_experiment_command — failure aggregation must raise RuntimeError
# ---------------------------------------------------------------------------


class _SyncExecutor:
    """In-process drop-in replacement for ProcessPoolExecutor.

    Runs ``submit`` synchronously and returns a completed Future. This lets
    us monkeypatch task functions and observe their behavior without paying
    the cost of real multiprocessing — and without the worker process
    bypassing our monkeypatch.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _SyncExecutor:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def submit(
        self,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[Any]:
        f: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f


class TestRunExperimentCommandFailures:
    """Verify Phase 4 expectation: failures bubble up as a RuntimeError."""

    def _write_phase2_cfg(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "p2.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "signal_types": ["ecg"],
                    "flicker_frequencies": [50.0],
                    "input_snrs": [10.0],
                    "training_epochs": [1],
                    "stdp_taus": [20.0],
                    "seeds_per_config": 2,
                    "duration_s": 1.0,
                }
            )
        )
        return cfg

    def test_all_failures_raise_runtime_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """If every task raises, run_experiment_command raises RuntimeError."""
        monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncExecutor)

        def _always_fail(task: dict[str, Any]) -> RunResult:
            raise ValueError("synthetic failure")

        monkeypatch.setattr(cli_mod, "run_single_experiment", _always_fail)

        cfg = self._write_phase2_cfg(tmp_path)
        args = argparse.Namespace(
            config=str(cfg),
            output_db=str(tmp_path / "out.db"),
            dry_run=False,
        )
        with pytest.raises(RuntimeError, match="runs failed"):
            run_experiment_command(args)

        out = capsys.readouterr().out
        assert "Error in run" in out

    def test_partial_failures_raise_runtime_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even one failing task among many causes RuntimeError."""
        call_count = {"n": 0}

        def _flaky(task: dict[str, Any]) -> RunResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first run boom")
            return RunResult(
                signal_type=task["signal_type"],
                flicker_freq=task["flicker_freq"],
                input_snr=task["input_snr"],
                epochs=task["epochs"],
                stdp_tau=task["stdp_tau"],
                seed=task["seed"],
                delta_snr=1.0,
                convergence_speed=1,
            )

        monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SyncExecutor)
        monkeypatch.setattr(cli_mod, "run_single_experiment", _flaky)

        cfg = self._write_phase2_cfg(tmp_path)
        db_path = tmp_path / "out.db"
        args = argparse.Namespace(
            config=str(cfg),
            output_db=str(db_path),
            dry_run=False,
        )
        with pytest.raises(RuntimeError, match="1 / 2 runs failed"):
            run_experiment_command(args)

        # The non-failing task should still have inserted its row.
        conn = sqlite3.connect(str(db_path))
        n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()
        assert n == 1
