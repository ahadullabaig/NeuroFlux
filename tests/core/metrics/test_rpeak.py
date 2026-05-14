"""Tests for R-peak integrity metric."""

from __future__ import annotations

import numpy as np
import pytest
from neuroflux_core.metrics.rpeak import rpeak_integrity


class TestRpeakIntegrity:
    """Tests for rpeak_integrity scoring."""

    def _make_ecg_like(
        self,
        duration_s: float = 5.0,
        fs: float = 250.0,
        heart_rate: float = 60.0,
    ) -> np.ndarray:
        """Generate a simple ECG-like signal with clear R-peaks."""
        t = np.arange(int(duration_s * fs)) / fs
        beat_interval = 60.0 / heart_rate
        beat_t = (t % beat_interval) - (0.5 * beat_interval)

        # QRS-like morphology via Gaussian mixture
        signal = np.zeros_like(t)
        signal += 1.0 * np.exp(-(beat_t**2) / (2 * 0.015**2))  # R wave
        signal += -0.25 * np.exp(-((beat_t - 0.05) ** 2) / (2 * 0.01**2))  # S wave
        signal += 0.15 * np.exp(-((beat_t - 0.3) ** 2) / (2 * 0.04**2))  # T wave
        return signal

    def test_perfect_reconstruction_returns_one(self) -> None:
        """Identical signals → score 1.0."""
        sig = self._make_ecg_like()
        assert rpeak_integrity(sig, sig.copy(), fs=250.0) == 1.0

    def test_flat_signal_returns_zero(self) -> None:
        """Flat (zero) signal has no R-peaks → score 0.0."""
        flat = np.zeros(1000)
        assert rpeak_integrity(flat, flat.copy(), fs=250.0) == 0.0

    def test_flat_reconstruction_returns_zero(self) -> None:
        """Clean has peaks but reconstruction is flat → score 0.0."""
        sig = self._make_ecg_like()
        flat = np.zeros_like(sig)
        assert rpeak_integrity(sig, flat, fs=250.0) == 0.0

    def test_small_shift_within_tolerance(self) -> None:
        """Reconstruction shifted by < tol_ms still matches."""
        sig = self._make_ecg_like(duration_s=10.0)
        # Shift by 5 samples at 250 Hz = 20 ms (within default 50 ms tol)
        shifted = np.roll(sig, 5)
        # Zero out edges to avoid wrap-around artifacts
        shifted[:10] = 0.0

        score = rpeak_integrity(sig, shifted, fs=250.0, tol_ms=50.0)
        # Most peaks should still match (edge effects may lose 1)
        assert score >= 0.8

    def test_large_shift_outside_tolerance(self) -> None:
        """Reconstruction shifted far beyond tolerance → low score."""
        sig = self._make_ecg_like(duration_s=10.0)
        # Shift by 50 samples at 250 Hz = 200 ms (well beyond 50 ms tol)
        shifted = np.roll(sig, 50)
        shifted[:60] = 0.0

        score = rpeak_integrity(sig, shifted, fs=250.0, tol_ms=50.0)
        assert score < 0.5

    def test_score_between_zero_and_one(self) -> None:
        """Score is always in [0, 1]."""
        sig = self._make_ecg_like()
        noisy = sig + np.random.default_rng(42).normal(0, 0.3, len(sig))

        score = rpeak_integrity(sig, noisy, fs=250.0)
        assert 0.0 <= score <= 1.0

    def test_shape_mismatch_raises(self) -> None:
        """Different-length signals raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            rpeak_integrity(np.zeros(100), np.zeros(200), fs=250.0)

    def test_tighter_tolerance_lowers_score(self) -> None:
        """Tighter tolerance should produce equal or lower scores."""
        sig = self._make_ecg_like(duration_s=10.0)
        shifted = np.roll(sig, 3)
        shifted[:5] = 0.0

        score_loose = rpeak_integrity(sig, shifted, fs=250.0, tol_ms=100.0)
        score_tight = rpeak_integrity(sig, shifted, fs=250.0, tol_ms=5.0)

        assert score_tight <= score_loose

    def test_noisy_reconstruction_partial_match(self) -> None:
        """Moderately noisy reconstruction gets a partial score."""
        sig = self._make_ecg_like(duration_s=10.0)
        rng = np.random.default_rng(99)
        noisy = sig + rng.normal(0, 0.5, len(sig))

        score = rpeak_integrity(sig, noisy, fs=250.0)
        # Should recover some but not necessarily all peaks
        assert 0.0 < score <= 1.0
