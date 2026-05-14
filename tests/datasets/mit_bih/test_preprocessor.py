"""Tests for MIT-BIH signal preprocessor: resampling and normalization."""

from __future__ import annotations

import numpy as np
import pytest
from neuroflux_datasets.mit_bih.preprocessor import normalize, resample_signal

# -----------------------------------------------------------------------
# resample_signal
# -----------------------------------------------------------------------


class TestResampleSignal:
    """Tests for polyphase resampling (360 → 250 Hz default)."""

    def test_360_to_250_output_length_1d(self) -> None:
        """10 s at 360 Hz → exactly 2500 samples at 250 Hz."""
        sig = np.zeros(3600)  # 10 s × 360 Hz
        out = resample_signal(sig)
        assert out.shape == (2500,)

    def test_360_to_250_output_length_2d(self) -> None:
        """Multi-channel: (3600, 2) → (2500, 2)."""
        sig = np.zeros((3600, 2))
        out = resample_signal(sig)
        assert out.shape == (2500, 2)

    def test_preserves_low_frequency_sine(self) -> None:
        """A 5 Hz sine at 360 Hz keeps its peak frequency after 250 Hz resample."""
        fs_in = 360.0
        fs_out = 250.0
        duration = 2.0  # seconds
        freq_hz = 5.0

        t_in = np.arange(int(duration * fs_in)) / fs_in
        sig = np.sin(2 * np.pi * freq_hz * t_in)

        out = resample_signal(sig, fs_in=fs_in, fs_out=fs_out)

        # FFT of output — peak should be at 5 Hz
        n = len(out)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_out)
        spectrum = np.abs(np.fft.rfft(out))
        peak_freq = freqs[np.argmax(spectrum[1:]) + 1]  # skip DC

        assert abs(peak_freq - freq_hz) < 1.0  # within 1 Hz bin

    def test_same_rate_returns_copy(self) -> None:
        """fs_in == fs_out returns an identical-value copy, not the same object."""
        sig = np.array([1.0, 2.0, 3.0])
        out = resample_signal(sig, fs_in=250.0, fs_out=250.0)

        np.testing.assert_array_equal(out, sig)
        assert out is not sig  # must be a copy

    def test_custom_rates(self) -> None:
        """Resampling with non-default rates produces expected length."""
        # 500 → 250 Hz should halve the length
        sig = np.ones(1000)
        out = resample_signal(sig, fs_in=500.0, fs_out=250.0)
        assert out.shape == (500,)

    def test_short_signal(self) -> None:
        """Very short signal (< 1 period) still resamples without error."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = resample_signal(sig, fs_in=360.0, fs_out=250.0)
        # 5 * 25/36 ≈ 3.47 → resample_poly rounds, expect 3 or 4
        assert len(out) > 0

    def test_invalid_fs_raises(self) -> None:
        """Zero or negative sampling rate raises ValueError."""
        sig = np.ones(100)
        with pytest.raises(ValueError, match="positive"):
            resample_signal(sig, fs_in=0.0, fs_out=250.0)
        with pytest.raises(ValueError, match="positive"):
            resample_signal(sig, fs_in=360.0, fs_out=-1.0)


# -----------------------------------------------------------------------
# normalize
# -----------------------------------------------------------------------


class TestNormalize:
    """Tests for per-channel zero-mean unit-variance normalization."""

    def test_1d_mean_near_zero(self) -> None:
        """Normalized 1-D signal has mean ≈ 0."""
        sig = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        out = normalize(sig)
        assert abs(float(np.mean(out))) < 1e-10

    def test_1d_std_near_one(self) -> None:
        """Normalized 1-D signal has std ≈ 1."""
        sig = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        out = normalize(sig)
        assert abs(float(np.std(out)) - 1.0) < 1e-10

    def test_1d_output_dtype_float64(self) -> None:
        """Output is always float64 regardless of input dtype."""
        sig = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        out = normalize(sig)
        assert out.dtype == np.float64

    def test_2d_per_channel_mean(self) -> None:
        """Each channel of a 2-D signal has mean ≈ 0 after normalization."""
        rng = np.random.default_rng(42)
        sig = rng.normal(loc=[100.0, -50.0], scale=[5.0, 20.0], size=(1000, 2))
        out = normalize(sig)

        for ch in range(2):
            assert abs(float(np.mean(out[:, ch]))) < 1e-10

    def test_2d_per_channel_std(self) -> None:
        """Each channel of a 2-D signal has std ≈ 1 after normalization."""
        rng = np.random.default_rng(42)
        sig = rng.normal(loc=[100.0, -50.0], scale=[5.0, 20.0], size=(1000, 2))
        out = normalize(sig)

        for ch in range(2):
            assert abs(float(np.std(out[:, ch])) - 1.0) < 1e-10

    def test_zero_variance_1d_returns_zeros(self) -> None:
        """A constant 1-D signal normalizes to all zeros."""
        sig = np.full(100, 42.0)
        out = normalize(sig)

        np.testing.assert_array_equal(out, np.zeros(100))

    def test_zero_variance_2d_mixed(self) -> None:
        """Constant channel → zeros, varying channel → normalized."""
        sig = np.column_stack(
            [
                np.full(200, 7.0),  # constant → zeros
                np.arange(200, dtype=float),  # varying → normalized
            ]
        )
        out = normalize(sig)

        # Constant channel is all zeros
        np.testing.assert_array_equal(out[:, 0], np.zeros(200))

        # Varying channel has mean ≈ 0, std ≈ 1
        assert abs(float(np.mean(out[:, 1]))) < 1e-10
        assert abs(float(np.std(out[:, 1])) - 1.0) < 1e-10

    def test_does_not_modify_input(self) -> None:
        """Normalize returns a new array; the input is unchanged."""
        sig = np.array([10.0, 20.0, 30.0])
        original = sig.copy()
        _ = normalize(sig)
        np.testing.assert_array_equal(sig, original)

    def test_preserves_shape(self) -> None:
        """Output shape matches input shape for both 1-D and 2-D."""
        sig_1d = np.arange(50, dtype=float)
        sig_2d = np.arange(100, dtype=float).reshape(50, 2)

        assert normalize(sig_1d).shape == (50,)
        assert normalize(sig_2d).shape == (50, 2)
