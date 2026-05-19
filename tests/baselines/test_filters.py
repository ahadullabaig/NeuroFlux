"""Unit tests for classical baseline filters.

Synthetic-signal tests only — no MIT-BIH data required, so these run in
``make test-fast`` on every push.
"""

from __future__ import annotations

import numpy as np
import pytest
from neuroflux_baselines import butterworth_notch, lms


class TestButterworthNotch:
    def test_attenuates_target_frequency(self) -> None:
        """50 Hz notch must suppress 50 Hz while preserving 5 Hz content."""
        fs = 250.0
        duration_s = 4.0
        t = np.arange(0, duration_s, 1.0 / fs)
        signal_5hz = np.sin(2 * np.pi * 5.0 * t)
        signal_50hz = np.sin(2 * np.pi * 50.0 * t)
        combined = signal_5hz + signal_50hz

        filtered = butterworth_notch(combined, freq_hz=50.0, fs=fs)

        # 5 Hz component should survive within 10% of its input amplitude
        # (filtfilt boundary effects can shave a small amount at the edges).
        signal_5_power = float(np.var(signal_5hz))
        residual_5_power = float(np.var(filtered - signal_5hz))
        assert residual_5_power < 0.10 * signal_5_power, (
            f"5 Hz content not preserved: residual_power={residual_5_power:.4f}, "
            f"input_power={signal_5_power:.4f}"
        )

        # 50 Hz component must be heavily attenuated.
        in_50_power = float(np.var(signal_50hz))
        out_50_power = float(
            np.var(filtered - signal_5hz)
        )  # what's left after removing 5 Hz
        assert out_50_power < 0.05 * in_50_power, (
            f"50 Hz not attenuated enough: out={out_50_power:.4f}, in={in_50_power:.4f}"
        )

    def test_rejects_dc_frequency(self) -> None:
        with pytest.raises(ValueError):
            butterworth_notch(np.zeros(100), freq_hz=0.0, fs=250.0)

    def test_rejects_frequency_above_nyquist(self) -> None:
        with pytest.raises(ValueError):
            butterworth_notch(np.zeros(100), freq_hz=200.0, fs=250.0)

    def test_rejects_2d_signal(self) -> None:
        with pytest.raises(ValueError):
            butterworth_notch(np.zeros((10, 10)), freq_hz=50.0, fs=250.0)

    def test_preserves_dtype(self) -> None:
        signal = np.random.default_rng(0).standard_normal(500).astype(np.float64)
        filtered = butterworth_notch(signal, freq_hz=50.0, fs=250.0)
        assert filtered.dtype == signal.dtype
        assert filtered.shape == signal.shape


class TestLMS:
    def test_reduces_noise_variance(self) -> None:
        """LMS on a low-frequency signal + white noise should reduce variance.

        Self-referencing LMS predicts each sample from the recent past, so the
        prediction error contains less of the predictable low-frequency content
        and proportionally more of the noise — the output variance is bounded
        above by the noise variance of the input. We assert the prediction
        error has lower total variance than the input.
        """
        rng = np.random.default_rng(0)
        fs = 250.0
        t = np.arange(0, 4.0, 1.0 / fs)
        clean = np.sin(2 * np.pi * 3.0 * t)
        noise = 0.3 * rng.standard_normal(len(t))
        noisy = clean + noise

        output = lms(noisy, filter_order=16)

        # The LMS prediction-error output should have lower total variance
        # than the noisy input once the predictor adapts (skip the warm-up).
        warmup = 500
        assert np.var(output[warmup:]) < np.var(noisy[warmup:]), (
            f"LMS did not reduce variance: out={np.var(output[warmup:]):.4f}, "
            f"in={np.var(noisy[warmup:]):.4f}"
        )

    def test_rejects_invalid_order(self) -> None:
        with pytest.raises(ValueError):
            lms(np.zeros(100), filter_order=0)

    def test_rejects_invalid_mu(self) -> None:
        with pytest.raises(ValueError):
            lms(np.zeros(100), mu=0.0)

    def test_rejects_2d_signal(self) -> None:
        with pytest.raises(ValueError):
            lms(np.zeros((10, 10)))

    def test_output_shape_matches_input(self) -> None:
        signal = np.random.default_rng(0).standard_normal(500)
        output = lms(signal)
        assert output.shape == signal.shape
