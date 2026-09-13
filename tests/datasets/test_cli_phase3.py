"""Tests for the Phase 3 CLI integration glue in ``neuroflux_datasets.cli``.

The MIT-BIH data-layer modules are tested in isolation under
``tests/datasets/mit_bih/``.  This module covers the *integration* layer
that wires real-dataset tasks through the experiment runner:

- ``_resolve_signal`` dispatch (synthetic vs. mit-bih branches)
- ``_build_phase3_tasks`` / ``_build_tasks`` config dispatch
- ``run_single_experiment`` end-to-end with ``dataset="mit-bih"``,
  including the ``rpeak_integrity`` wiring
- ``--dry-run`` opens no database

All ``wfdb`` / network access is avoided by monkeypatching ``make_pair``
at its source module, so these tests never download anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neuroflux_datasets.cli import (
    Phase3Config,
    _build_phase3_tasks,
    _build_tasks,
    _resolve_reference_beats,
    _resolve_signal,
    run_experiment_command,
    run_single_experiment,
)
from neuroflux_datasets.mit_bih.annotator import AnnotatedSegment

_FS = 250.0


def _make_ecg_pair(duration_s: float, fs: float = _FS) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic ``(clean, noisy)`` ECG-like pair.

    Length is exactly ``int(duration_s * fs)`` so it matches the
    ``n_steps`` the Simulator produces for the same duration — this is
    what lets ``rpeak_integrity(clean, reconstructed)`` line up.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    beat_t = (t % 1.0) - 0.5
    clean = np.exp(-(beat_t**2) / (2 * 0.015**2))  # clear R-peaks @ 60 bpm
    noisy = clean + np.random.default_rng(0).normal(0, 0.2, n)
    return clean, noisy


@pytest.fixture()
def patched_make_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the annotator so no record files are touched.

    Both ``make_pair`` (the signal) and ``extract_annotated_segments`` (the
    reference beats) are replaced. Patching only the first would leave the QRS
    scoring reading real ``.atr`` files — which passes on a machine with cached
    records and fails in CI without them.
    """

    def _fake_make_pair(
        *, duration_s: float, fs: float = _FS, **_: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        return _make_ecg_pair(duration_s, fs)

    def _fake_annotated(
        record_id: str, cache_dir: Any, duration_s: float, fs: float = _FS, **_: Any
    ) -> list[AnnotatedSegment]:
        clean, _noisy = _make_ecg_pair(duration_s, fs)
        # `_make_ecg_pair` peaks half-way through every one-second beat.
        beats = np.arange(int(0.5 * fs), clean.size, int(fs), dtype=np.int64)
        return [AnnotatedSegment(signal=clean, beats=beats, start_sample=0)]

    monkeypatch.setattr(
        "neuroflux_datasets.mit_bih.annotator.make_pair", _fake_make_pair
    )
    monkeypatch.setattr(
        "neuroflux_datasets.mit_bih.annotator.extract_annotated_segments",
        _fake_annotated,
    )


# ---------------------------------------------------------------------------
# _resolve_signal — dispatch
# ---------------------------------------------------------------------------


class TestResolveSignal:
    """Verify _resolve_signal dispatches correctly on the ``dataset`` key."""

    def test_synthetic_branch_is_default(self) -> None:
        """No ``dataset`` key → synthetic path (Phase 2 behaviour)."""
        params = {
            "signal_type": "ecg",
            "duration_s": 1.0,
            "flicker_freq": 50.0,
            "input_snr": 10.0,
            "seed": 0,
        }
        clean, noisy = _resolve_signal(params, _FS)
        assert clean.shape == noisy.shape
        assert clean.shape == (int(1.0 * _FS),)

    def test_synthetic_branch_is_deterministic(self) -> None:
        """Same params → identical signals (no wfdb import involved)."""
        params = {
            "signal_type": "ecg",
            "duration_s": 1.0,
            "flicker_freq": 50.0,
            "input_snr": 10.0,
            "seed": 7,
        }
        c1, n1 = _resolve_signal(params, _FS)
        c2, n2 = _resolve_signal(params, _FS)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(n1, n2)

    def test_mit_bih_branch_calls_make_pair(self, patched_make_pair: None) -> None:
        """``dataset="mit-bih"`` routes through make_pair with task params."""
        params = {
            "dataset": "mit-bih",
            "record_id": "100",
            "cache_dir": "data/mit_bih",
            "duration_s": 2.0,
            "flicker_freq": 50.0,
            "flicker_amplitude": 0.5,
            "input_snr": 10.0,
            "seed": 0,
        }
        clean, noisy = _resolve_signal(params, _FS)
        assert clean.shape == noisy.shape
        assert clean.shape == (int(2.0 * _FS),)


# ---------------------------------------------------------------------------
# _build_tasks / _build_phase3_tasks — config dispatch
# ---------------------------------------------------------------------------


class TestBuildTasks:
    """Verify task-dict construction for both config types."""

    def test_phase3_task_count_is_cartesian_product(self) -> None:
        """Phase 3 task count = product of all swept axes × seeds."""
        config = Phase3Config(
            record_ids=["100", "101", "103"],
            flicker_frequencies=[50.0, 60.0],
            input_snrs=[10.0],
            training_epochs=[5],
            stdp_taus=[20.0],
            seeds_per_config=2,
            duration_s=5.0,
        )
        tasks = _build_phase3_tasks(config)
        assert len(tasks) == 3 * 2 * 1 * 1 * 1 * 2

    def test_phase3_tasks_carry_required_keys(self) -> None:
        """Every Phase 3 task dict has the keys run_single_experiment needs."""
        config = Phase3Config(
            record_ids=["100"],
            flicker_frequencies=[50.0],
            input_snrs=[10.0],
            training_epochs=[5],
            stdp_taus=[20.0],
            seeds_per_config=1,
        )
        task = _build_phase3_tasks(config)[0]
        assert task["dataset"] == "mit-bih"
        assert task["signal_type"] == "ecg"  # for analysis.py grouping
        assert task["record_id"] == "100"
        assert task["cache_dir"] == config.cache_dir
        for key in ("flicker_freq", "input_snr", "epochs", "stdp_tau", "seed"):
            assert key in task

    def test_build_tasks_dispatches_on_dataset_key(self) -> None:
        """Raw YAML with ``dataset: mit-bih`` builds Phase 3 tasks."""
        raw = {
            "dataset": "mit-bih",
            "record_ids": ["100", "101"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [5],
            "stdp_taus": [20.0],
            "seeds_per_config": 1,
            "duration_s": 5.0,
        }
        tasks = _build_tasks(raw)
        assert len(tasks) == 2
        assert all(t["dataset"] == "mit-bih" for t in tasks)

    def test_build_tasks_defaults_to_phase2(self) -> None:
        """Raw YAML with no ``dataset`` key builds Phase 2 synthetic tasks."""
        raw = {
            "signal_types": ["ecg"],
            "flicker_frequencies": [50.0],
            "input_snrs": [10.0],
            "training_epochs": [50],
            "stdp_taus": [20.0],
            "seeds_per_config": 2,
        }
        tasks = _build_tasks(raw)
        assert len(tasks) == 2
        assert all("dataset" not in t for t in tasks)


# ---------------------------------------------------------------------------
# run_single_experiment — end-to-end with dataset="mit-bih"
# ---------------------------------------------------------------------------


class TestRunSingleExperimentMitBih:
    """Verify a full mit-bih task runs through the SNN pipeline."""

    def _mit_bih_params(self) -> dict[str, Any]:
        return {
            "dataset": "mit-bih",
            "signal_type": "ecg",
            "record_id": "100",
            # Deliberately absent, so any unpatched file access fails here
            # rather than silently passing on a machine with cached records.
            "cache_dir": "/nonexistent/mit_bih",
            "flicker_freq": 50.0,
            "flicker_amplitude": 0.5,
            "input_snr": 10.0,
            "epochs": 1,
            "stdp_tau": 20.0,
            "seed": 0,
            "duration_s": 4.0,
        }

    def test_returns_runresult_tagged_mit_bih(self, patched_make_pair: None) -> None:
        """RunResult carries dataset/record_id from the mit-bih task."""
        res = run_single_experiment(self._mit_bih_params())
        assert res.dataset == "mit-bih"
        assert res.record_id == "100"
        assert res.signal_type == "ecg"

    def test_rpeak_integrity_is_populated(self, patched_make_pair: None) -> None:
        """rpeak_integrity is computed (non-None, in [0, 1]) for mit-bih."""
        res = run_single_experiment(self._mit_bih_params())
        assert res.rpeak_integrity is not None
        assert 0.0 <= res.rpeak_integrity <= 1.0

    def test_qrs_scores_are_populated(self, patched_make_pair: None) -> None:
        """The R1 QRS columns are computed for mit-bih, and bounded."""
        res = run_single_experiment(self._mit_bih_params())
        for value in (res.qrs_f1, res.qrs_sensitivity, res.qrs_ppv):
            assert value is not None
            assert 0.0 <= value <= 1.0

    def test_unmatched_jitter_is_stored_as_none_not_nan(
        self, patched_make_pair: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SQLite has no NaN, so a window with no matches must carry None."""
        monkeypatch.setattr(
            "neuroflux_core.metrics.qrs.detect_rpeaks",
            lambda *_, **__: np.empty(0, dtype=np.int64),
        )
        res = run_single_experiment(self._mit_bih_params())
        assert res.qrs_f1 == 0.0
        assert res.qrs_jitter_ms is None

    def test_a_record_without_clean_windows_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The beat lookup fails loudly rather than scoring an empty reference."""
        monkeypatch.setattr(
            "neuroflux_datasets.mit_bih.annotator.extract_annotated_segments",
            lambda *_, **__: [],
        )
        with pytest.raises(ValueError, match="no artifact-free segment"):
            _resolve_reference_beats(self._mit_bih_params(), _FS)

    def test_synthetic_run_leaves_rpeak_none(self) -> None:
        """The synthetic path must NOT compute rpeak_integrity."""
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
        assert res.dataset == "synthetic"
        assert res.record_id is None
        assert res.rpeak_integrity is None
        assert res.qrs_f1 is None
        assert res.qrs_jitter_ms is None


# ---------------------------------------------------------------------------
# --dry-run — must not open a database
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify --dry-run validates without side effects."""

    def test_dry_run_creates_no_database(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Phase 3 dry run prints task count and writes no DB file."""
        config_path = tmp_path / "phase3.yaml"
        config_path.write_text(
            "dataset: mit-bih\n"
            'record_ids: ["100", "101"]\n'
            "flicker_frequencies: [50.0]\n"
            "input_snrs: [10.0]\n"
            "training_epochs: [5]\n"
            "stdp_taus: [20.0]\n"
            "seeds_per_config: 1\n"
            "duration_s: 5.0\n"
        )
        db_path = tmp_path / "should_not_exist.db"
        args = argparse.Namespace(
            config=str(config_path),
            output_db=str(db_path),
            dry_run=True,
        )

        run_experiment_command(args)

        assert not db_path.exists()
        out = capsys.readouterr().out
        assert "Dry run: 2 tasks validated." in out
        assert "mit-bih" in out
