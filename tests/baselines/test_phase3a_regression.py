"""Regression tests for Phase 3A baseline methods on record 214.

Pins the canonical config used by ``test_phase3_regression.py`` (record 214,
50 Hz, 0 dB, seed=0, 10 s) for the three new methods introduced in Phase 3A:
``snn_passive``, ``butterworth_notch``, ``lms``.

**Thresholds are pinned in a follow-up commit** after the Phase 3A full sweep
runs and produces actual numbers in ``results/phase3a/method_aggregates.csv``.
Until then, each test calls ``run_single_experiment`` to exercise the code
path (catching wiring regressions) but uses ``pytest.skip`` to defer the
numerical assertion.  This keeps the test file honest: it exists and runs the
new dispatch paths, but it does not claim a number that hasn't been measured.

Like the Phase 3 regression test, this skips cleanly if record 214 isn't
cached locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from neuroflux_datasets.cli import run_single_experiment

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / "data" / "mit_bih"
_RECORD_DAT = _CACHE_DIR / "214.dat"

pytestmark = pytest.mark.skipif(
    not _RECORD_DAT.exists(),
    reason=(
        "MIT-BIH record 214 not cached locally. "
        'Run `uv run python -c "from neuroflux_datasets.mit_bih.downloader '
        "import download_record; download_record('214', 'data/mit_bih')\"` "
        "to enable Phase 3A regression tests."
    ),
)


def _canonical_params(
    method: str, *, epochs: int = 0, stdp_tau: float = 10.0
) -> dict[str, Any]:
    return {
        "dataset": "mit-bih",
        "signal_type": "ecg",
        "method": method,
        "record_id": "214",
        "cache_dir": str(_CACHE_DIR),
        "flicker_freq": 50.0,
        "flicker_amplitude": 0.5,
        "input_snr": 0.0,
        "epochs": epochs,
        "stdp_tau": stdp_tau,
        "seed": 0,
        "duration_s": 10.0,
    }


def test_snn_passive_record_214_runs_and_preserves_rpeaks() -> None:
    """Passive SNN (epochs=0) must hit rpeak ≈ 1.000 by §5.6.2 of findings.

    Threshold tightened in follow-up after Phase 3A sweep measures the
    actual seed-0 rpeak.  For now, sanity-check that the wiring works and
    rpeak is meaningfully positive — anything below 0.50 would indicate
    the passive-path encoder/decoder is broken.
    """
    result = run_single_experiment(_canonical_params("snn_passive"))
    assert result.method == "snn_passive"
    assert result.epochs == 0
    assert result.rpeak_integrity is not None
    assert result.rpeak_integrity > 0.50, (
        f"passive SNN rpeak collapsed to {result.rpeak_integrity:.3f} — "
        "encoder/decoder pipeline regression?"
    )
    pytest.skip(
        "Numerical thresholds for snn_passive pinned in follow-up after the "
        "Phase 3A sweep produces results/phase3a/method_aggregates.csv."
    )


def test_butterworth_notch_record_214_runs() -> None:
    """Butterworth notch must run end-to-end with non-NaN metrics."""
    result = run_single_experiment(_canonical_params("butterworth_notch"))
    assert result.method == "butterworth_notch"
    assert result.convergence_speed == 0
    assert result.rpeak_integrity is not None
    # Sanity: delta_snr finite, rpeak in [0, 1].
    assert -50.0 < result.delta_snr < 50.0
    assert 0.0 <= result.rpeak_integrity <= 1.0
    pytest.skip(
        "Numerical thresholds for butterworth_notch pinned in follow-up "
        "after the Phase 3A sweep produces results/phase3a/method_aggregates.csv."
    )


def test_lms_record_214_runs() -> None:
    """LMS must run end-to-end with non-NaN metrics."""
    result = run_single_experiment(_canonical_params("lms"))
    assert result.method == "lms"
    assert result.convergence_speed == 0
    assert result.rpeak_integrity is not None
    assert -50.0 < result.delta_snr < 50.0
    assert 0.0 <= result.rpeak_integrity <= 1.0
    pytest.skip(
        "Numerical thresholds for lms pinned in follow-up after the "
        "Phase 3A sweep produces results/phase3a/method_aggregates.csv."
    )
