"""Tests for affine-corrected SNR and the degeneracy guard.

These two are load-bearing for every ceiling number in the paper, so the
saturation test at the bottom is the important one: the metric must separate
a real reconstruction from the identity baseline.
"""

import numpy as np
import pytest
from neuroflux_core.metrics.degeneracy import (
    DEFAULT_MIN_STD,
    is_degenerate,
    output_std,
)
from neuroflux_core.metrics.reconstruction import affine_corrected_snr, affine_fit
from neuroflux_core.metrics.snr import calculate_snr


def _clean(n: int = 500) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return np.sin(t) + 0.3 * np.sin(3.0 * t)


# --------------------------------------------------------------------------
# affine_fit
# --------------------------------------------------------------------------


def test_affine_fit_recovers_known_transform() -> None:
    clean = _clean()
    scaled = 0.25 * clean + 3.0  # invert: a = 4.0, b = -12.0
    scale, offset = affine_fit(clean, scaled)
    assert scale == pytest.approx(4.0, abs=1e-6)
    assert offset == pytest.approx(-12.0, abs=1e-6)


def test_affine_fit_identity_on_itself() -> None:
    clean = _clean()
    scale, offset = affine_fit(clean, clean)
    assert scale == pytest.approx(1.0, abs=1e-9)
    assert offset == pytest.approx(0.0, abs=1e-9)


def test_affine_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        affine_fit(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="Length mismatch"):
        affine_fit(np.zeros(10), np.zeros(5))


# --------------------------------------------------------------------------
# affine_corrected_snr
# --------------------------------------------------------------------------


def test_affine_correction_is_invariant_to_scale_and_offset() -> None:
    """The whole point: a rescaled reconstruction must not be penalised."""
    clean = _clean()
    rng = np.random.default_rng(0)
    recon = clean + rng.normal(0.0, 0.1, size=clean.size)

    plain = affine_corrected_snr(clean, recon)
    rescaled = affine_corrected_snr(clean, 17.0 * recon - 4.0)
    assert rescaled == pytest.approx(plain, abs=1e-6)


def test_affine_correction_beats_raw_snr_on_misscaled_output() -> None:
    clean = _clean()
    misscaled = 0.05 * clean + 10.0
    assert affine_corrected_snr(clean, misscaled) > calculate_snr(clean, misscaled)


def test_affine_correction_still_penalises_shape_error() -> None:
    """It must not become a free pass — a wrong shape still scores badly."""
    clean = _clean()
    rng = np.random.default_rng(1)
    good = clean + rng.normal(0.0, 0.05, size=clean.size)
    bad = clean + rng.normal(0.0, 0.80, size=clean.size)
    assert affine_corrected_snr(clean, good) > affine_corrected_snr(clean, bad) + 10.0


# --------------------------------------------------------------------------
# degeneracy
# --------------------------------------------------------------------------


def test_constant_and_empty_and_nonfinite_are_degenerate() -> None:
    assert is_degenerate(np.zeros(100))
    assert is_degenerate(np.full(100, 7.5))
    assert is_degenerate(np.array([]))
    assert is_degenerate(np.array([1.0, np.nan, 3.0]))
    assert is_degenerate(np.array([1.0, np.inf, 3.0]))


def test_real_signal_is_not_degenerate() -> None:
    assert not is_degenerate(_clean())


def test_min_std_threshold_is_respected() -> None:
    signal = np.array([0.0, 0.001, 0.0, 0.001] * 50)
    assert not is_degenerate(signal, min_std=1e-6)
    assert is_degenerate(signal, min_std=1.0)


def test_is_degenerate_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        is_degenerate(_clean(), min_std=-1.0)


def test_output_std_matches_numpy() -> None:
    signal = _clean()
    assert output_std(signal) == pytest.approx(float(np.std(signal)))
    assert output_std(np.full(10, 2.0)) == pytest.approx(0.0)


def test_default_min_std_is_a_numerical_floor() -> None:
    assert DEFAULT_MIN_STD > 0.0
    assert DEFAULT_MIN_STD < 1e-3


# --------------------------------------------------------------------------
# The saturation test the roadmap requires (R1)
# --------------------------------------------------------------------------


def test_metric_separates_processed_signal_from_identity_baseline() -> None:
    """
    The metric must give materially different scores to a real reconstruction
    and to the untouched noisy input, across the SNR range, by more than the
    seed-to-seed spread. This is what `rpeak_integrity` fails to do.
    """
    clean = _clean(1000)

    for input_snr_db in (0.0, 5.0, 10.0, 20.0):
        identity_scores: list[float] = []
        processed_scores: list[float] = []

        for seed in range(8):
            rng = np.random.default_rng(seed)
            noise_power = np.mean(clean**2) / (10 ** (input_snr_db / 10.0))
            noisy = clean + rng.normal(0.0, np.sqrt(noise_power), size=clean.size)
            # A crude but genuine denoiser: 5-point moving average.
            processed = np.convolve(noisy, np.ones(5) / 5.0, mode="same")

            identity_scores.append(affine_corrected_snr(clean, noisy))
            processed_scores.append(affine_corrected_snr(clean, processed))

        gap = float(np.mean(processed_scores)) - float(np.mean(identity_scores))
        spread = max(float(np.std(identity_scores)), float(np.std(processed_scores)))
        assert gap > spread, (
            f"at {input_snr_db} dB the metric cannot separate processed from "
            f"identity: gap={gap:.3f} dB, seed spread={spread:.3f} dB"
        )
