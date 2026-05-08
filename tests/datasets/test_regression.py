"""CI regression test — canonical ECG config must achieve ΔSNR ≥ 4 dB.

PLAN.md requirement: the canonical config must achieve ΔSNR ≥ 4 dB on the
reference synthetic ECG and run in CI on every commit (fast, not @pytest.mark.slow).

Canonical parameters (best config from full_sweep results):
  signal_type  : ecg
  flicker_freq : 50 Hz  (most common powerline frequency)
  input_snr    : 10 dB  (mid-range, representative)
  epochs       : 100
  stdp_tau     : 20 ms
  seed         : 0
"""

from neuroflux_datasets.cli import run_single_experiment

_CANONICAL_PARAMS: dict[str, object] = {
    "signal_type": "ecg",
    "flicker_freq": 50.0,
    "input_snr": 10.0,
    "epochs": 100,
    "stdp_tau": 20.0,
    "seed": 0,
    "duration_s": 1.0,
    "flicker_amplitude": 0.5,
}

_DELTA_SNR_THRESHOLD = 4.0


class TestCanonicalECGRegression:
    """Regression guard: canonical ECG config must achieve ΔSNR ≥ 4 dB."""

    def test_canonical_ecg_meets_threshold(self) -> None:
        """ΔSNR on the canonical ECG config must be ≥ 4 dB."""
        result = run_single_experiment(_CANONICAL_PARAMS)
        assert result.delta_snr >= _DELTA_SNR_THRESHOLD, (
            f"Canonical ECG config achieved ΔSNR={result.delta_snr:.3f} dB, "
            f"expected ≥ {_DELTA_SNR_THRESHOLD} dB. "
            "A change to LIF dynamics, STDP, encoders, or signal generation "
            "may have regressed performance."
        )

    def test_canonical_ecg_is_deterministic(self) -> None:
        """Same seed must produce identical ΔSNR across runs."""
        r1 = run_single_experiment(_CANONICAL_PARAMS)
        r2 = run_single_experiment(_CANONICAL_PARAMS)
        assert r1.delta_snr == r2.delta_snr, (
            f"Non-deterministic result: {r1.delta_snr:.6f} vs {r2.delta_snr:.6f}"
        )
