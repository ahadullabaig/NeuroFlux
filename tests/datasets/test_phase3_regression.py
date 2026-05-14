"""Regression test for Phase 3 on real MIT-BIH ECG.

Pins the most stable positive configuration identified by the full
Phase 3 sweep (see ``docs/phase3_findings.md`` §3.7):

- Record 214 (LBBB + PVCs, prominent wide QRS complexes)
- 50 Hz flicker, 0 dB input SNR, 10 epochs, τ⁺ = τ⁻ = 10 ms
- 10 s segment at 250 Hz

Observed: mean ΔSNR ≈ +2.26 dB and mean rpeak_integrity ≈ 0.985 over
five seeds.  The bar is set conservatively below those numbers to allow
for seed-to-seed variation while still catching any regression in the
core engine, encoder/decoder pipeline, or MIT-BIH preprocessing.

This test requires record 214 to be cached locally — CI runners
typically don't bundle MIT-BIH data, so the test skips cleanly if the
``.dat`` file isn't present. To populate the cache locally::

    uv run python -c "
    from neuroflux_datasets.mit_bih.downloader import download_record
    download_record('214', 'data/mit_bih')
    "
"""

from __future__ import annotations

from pathlib import Path

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
        "to enable this regression test."
    ),
)


def test_phase3_canonical_record_214_ecg_flicker_suppression() -> None:
    """Canonical Phase 3 config must hit ΔSNR ≥ 1.5 dB and rpeak ≥ 0.85.

    Tolerances are conservative relative to the sweep means (+2.26 dB,
    0.985) — they leave headroom for seed/Numba variation while keeping
    the test sensitive to genuine regressions.
    """
    params = {
        "dataset": "mit-bih",
        "signal_type": "ecg",
        "record_id": "214",
        "cache_dir": str(_CACHE_DIR),
        "flicker_freq": 50.0,
        "flicker_amplitude": 0.5,
        "input_snr": 0.0,
        "epochs": 10,
        "stdp_tau": 10.0,
        "seed": 0,
        "duration_s": 10.0,
    }

    result = run_single_experiment(params)

    assert result.dataset == "mit-bih"
    assert result.record_id == "214"
    assert result.delta_snr >= 1.5, (
        f"Expected ΔSNR ≥ 1.5 dB on record 214, got {result.delta_snr:.2f} dB"
    )
    assert result.rpeak_integrity is not None
    assert result.rpeak_integrity >= 0.85, (
        f"Expected R-peak integrity ≥ 0.85 on record 214, "
        f"got {result.rpeak_integrity:.3f}"
    )
