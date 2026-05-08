"""Regression tests for Phase 2 flicker suppression."""

from neuroflux_datasets.cli import run_single_experiment


def test_canonical_ecg_flicker_suppression() -> None:
    """
    Automated regression test: canonical config must achieve ΔSNR ≥ 4 dB
    on the reference synthetic ECG.
    """
    params = {
        "signal_type": "ecg",
        "flicker_freq": 50.0,
        "input_snr": 10.0,
        "epochs": 50,
        "stdp_tau": 20.0,
        "seed": 42,
        "duration_s": 2.0,
    }

    result = run_single_experiment(params)

    assert result.delta_snr >= 4.0, (
        f"Expected ΔSNR >= 4.0 dB, got {result.delta_snr:.2f} dB"
    )
