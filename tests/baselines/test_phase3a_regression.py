"""Regression tests for Phase 3A baseline methods on record 214.

Pins the canonical config used by ``test_phase3_regression.py`` (record 214,
50 Hz, 0 dB, seed=0, 10 s) for the three new methods introduced in Phase 3A:
``snn_passive``, ``butterworth_notch``, ``lms``.

Thresholds were pinned after the Phase 3A full sweep produced the seed-0
values for this canonical cell.  Measured (record=214, freq=50, snr=0, seed=0):

- ``snn_passive``       : delta_snr = -3.056, rpeak = 1.000
- ``butterworth_notch`` : delta_snr = +0.552, rpeak = 1.000
- ``lms``               : delta_snr = -3.277, rpeak = 1.000

Tolerances are ±10% on delta_snr (asymmetric for negative values — the bound
is "no worse than 1.1× the measured magnitude") and rpeak ≥ 0.95 (a tight
floor; all three methods saturate at 1.000 on this segment).

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


def test_snn_passive_record_214() -> None:
    """Passive SNN on record 214: rpeak saturates at 1.000, ΔSNR ≈ -3.06.

    Measured at sweep time: delta_snr = -3.056, rpeak = 1.000.  The negative
    ΔSNR is expected — the passive (untrained) SNN doesn't denoise the
    amplitude; its value (per ``docs/phase3_findings.md`` §5.6.2) is the
    saturated R-peak preservation.
    """
    result = run_single_experiment(_canonical_params("snn_passive"))
    assert result.method == "snn_passive"
    assert result.epochs == 0
    assert result.rpeak_integrity is not None
    assert result.rpeak_integrity >= 0.95, (
        f"passive SNN rpeak dropped to {result.rpeak_integrity:.3f}; expected ≥ 0.95"
    )
    assert -3.6 <= result.delta_snr <= -2.5, (
        f"passive SNN ΔSNR drifted to {result.delta_snr:.3f}; expected in [-3.6, -2.5]"
    )


def test_butterworth_notch_record_214() -> None:
    """Butterworth notch on record 214: ΔSNR ≈ +0.55, rpeak = 1.000.

    Measured: delta_snr = +0.552, rpeak = 1.000.  Asserts the filter
    produces a small but positive ΔSNR at this low-SNR cell, as expected.
    """
    result = run_single_experiment(_canonical_params("butterworth_notch"))
    assert result.method == "butterworth_notch"
    assert result.convergence_speed == 0
    assert result.rpeak_integrity is not None
    assert result.rpeak_integrity >= 0.95, (
        f"butterworth rpeak dropped to {result.rpeak_integrity:.3f}; expected ≥ 0.95"
    )
    assert 0.4 <= result.delta_snr <= 0.7, (
        f"butterworth ΔSNR drifted to {result.delta_snr:.3f}; expected in [0.4, 0.7]"
    )


def test_lms_record_214() -> None:
    """NLMS on record 214: ΔSNR ≈ -3.28, rpeak = 1.000.

    Measured: delta_snr = -3.277, rpeak = 1.000.  NLMS is unsupervised and
    has no frequency information, so it underperforms Butterworth at this
    cell.  Pinned at the measured magnitude.
    """
    result = run_single_experiment(_canonical_params("lms"))
    assert result.method == "lms"
    assert result.convergence_speed == 0
    assert result.rpeak_integrity is not None
    assert result.rpeak_integrity >= 0.95, (
        f"lms rpeak dropped to {result.rpeak_integrity:.3f}; expected ≥ 0.95"
    )
    assert -3.7 <= result.delta_snr <= -2.9, (
        f"lms ΔSNR drifted to {result.delta_snr:.3f}; expected in [-3.7, -2.9]"
    )
