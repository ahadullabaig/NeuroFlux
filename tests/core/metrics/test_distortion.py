"""Tests for the ECG-literature distortion metrics.

The tests are written as the properties that make each metric worth reporting
rather than as coverage of each branch: that PRD1 is baseline-independent and
PRD is not (which is the whole reason both exist), that PRD1 and the SNR are
the same statement in different units when the residual is centred, and that
the cross-correlation peak separates a wrong shape from a shifted one.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from neuroflux_core.metrics.distortion import (
    cross_correlation_peak,
    max_cross_correlation,
    prd,
    prd1,
    rmse,
)
from neuroflux_core.metrics.snr import calculate_snr

FS = 250.0


def _wave(n: int = 1000, offset: float = 0.0) -> np.ndarray:
    """A deterministic non-constant test signal."""
    return np.sin(np.linspace(0.0, 20.0, n)) + offset


class TestExactMatch:
    """A reconstruction equal to the reference scores perfectly."""

    def test_prd_is_zero(self) -> None:
        signal = _wave()
        assert prd(signal, signal) == pytest.approx(0.0)

    def test_prd1_is_zero(self) -> None:
        signal = _wave()
        assert prd1(signal, signal) == pytest.approx(0.0)

    def test_rmse_is_zero(self) -> None:
        signal = _wave()
        assert rmse(signal, signal) == pytest.approx(0.0)

    def test_cross_correlation_is_one_at_zero_lag(self) -> None:
        signal = _wave()
        peak, lag = cross_correlation_peak(signal, signal)
        assert peak == pytest.approx(1.0)
        assert lag == 0


class TestBaselineIndependence:
    """The reason PRD and PRD1 both exist, stated as a test."""

    def test_prd_moves_when_a_dc_offset_is_added(self) -> None:
        clean, recon = _wave(), _wave() + 0.1
        shifted = prd(clean + 5.0, recon + 5.0)
        assert prd(clean, recon) != pytest.approx(shifted, rel=1e-3)

    def test_prd1_does_not_move_when_a_dc_offset_is_added(self) -> None:
        clean, recon = _wave(), _wave() + 0.1
        assert prd1(clean, recon) == pytest.approx(prd1(clean + 5.0, recon + 5.0))

    def test_a_dc_offset_makes_prd_look_better(self) -> None:
        # The failure mode the metric choice guards against: raising the
        # baseline inflates the denominator and improves the score, with the
        # reconstruction error completely unchanged.
        clean, recon = _wave(), _wave() + 0.1
        assert prd(clean + 5.0, recon + 5.0) < prd(clean, recon)


class TestPrd1AgreesWithSnr:
    """PRD1 and the SNR are one statement in two unit systems."""

    @pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 20.0])
    def test_identity_holds_for_a_centred_residual(self, snr_db: float) -> None:
        rng = np.random.default_rng(0)
        clean = _wave(4000)
        noise = rng.standard_normal(clean.size)
        noise -= noise.mean()
        scale = math.sqrt(float(np.var(clean)) / 10.0 ** (snr_db / 10.0))
        recon = clean + scale * noise / float(np.std(noise))

        measured = calculate_snr(clean, recon)
        assert prd1(clean, recon) == pytest.approx(
            100.0 * 10.0 ** (-measured / 20.0), rel=1e-6
        )


class TestDegenerateReferences:
    """Zero-energy references, which real pipelines do produce."""

    def test_prd_is_infinite_when_the_reference_has_no_energy(self) -> None:
        zeros = np.zeros(100)
        assert math.isinf(prd(zeros, np.ones(100)))

    def test_prd_is_zero_when_both_are_empty_of_energy(self) -> None:
        zeros = np.zeros(100)
        assert prd(zeros, zeros) == 0.0

    def test_prd1_is_infinite_for_a_constant_reference(self) -> None:
        assert math.isinf(prd1(np.full(100, 3.0), _wave(100)))

    def test_prd1_is_zero_for_two_identical_constants(self) -> None:
        constant = np.full(100, 3.0)
        assert prd1(constant, constant) == 0.0

    def test_cross_correlation_is_zero_for_a_constant_signal(self) -> None:
        assert cross_correlation_peak(_wave(100), np.zeros(100)) == (0.0, 0)


class TestCrossCorrelationLag:
    """A high peak at a nonzero lag means "right shape, wrong timing"."""

    def test_a_late_reconstruction_reports_a_positive_lag(self) -> None:
        rng = np.random.default_rng(1)
        clean = rng.standard_normal(500)
        late = np.zeros(500)
        late[7:] = clean[:-7]
        peak, lag = cross_correlation_peak(clean, late, max_lag_samples=30)
        assert lag == 7
        assert peak > 0.98

    def test_an_early_reconstruction_reports_a_negative_lag(self) -> None:
        rng = np.random.default_rng(1)
        clean = rng.standard_normal(500)
        early = np.zeros(500)
        early[:-7] = clean[7:]
        _, lag = cross_correlation_peak(clean, early, max_lag_samples=30)
        assert lag == -7

    def test_the_lag_search_respects_its_cap(self) -> None:
        rng = np.random.default_rng(1)
        clean = rng.standard_normal(500)
        late = np.zeros(500)
        late[20:] = clean[:-20]
        _, lag = cross_correlation_peak(clean, late, max_lag_samples=5)
        assert abs(lag) <= 5

    def test_a_shift_scores_worse_when_the_search_cannot_reach_it(self) -> None:
        rng = np.random.default_rng(1)
        clean = rng.standard_normal(500)
        late = np.zeros(500)
        late[20:] = clean[:-20]
        assert max_cross_correlation(clean, late, 5) < max_cross_correlation(
            clean, late, 30
        )

    def test_a_wrong_shape_scores_low_at_every_lag(self) -> None:
        rng = np.random.default_rng(2)
        clean = _wave(500)
        assert max_cross_correlation(clean, rng.standard_normal(500)) < 0.5


class TestValidation:
    """Bad inputs fail loudly rather than producing a plausible number."""

    @pytest.mark.parametrize("metric", [prd, prd1, rmse, max_cross_correlation])
    def test_length_mismatch_raises(self, metric: object) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            metric(np.ones(10), np.ones(11))  # type: ignore[operator]

    @pytest.mark.parametrize("metric", [prd, prd1, rmse, max_cross_correlation])
    def test_empty_input_raises(self, metric: object) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            metric(np.array([]), np.array([]))  # type: ignore[operator]

    def test_negative_max_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            max_cross_correlation(_wave(100), _wave(100), -1)


class TestProperties:
    """Bounds that must hold for any input."""

    @given(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=8,
            max_size=64,
        ),
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=8,
            max_size=64,
        ),
    )
    def test_cross_correlation_stays_within_unit_bounds(
        self, left: list[float], right: list[float]
    ) -> None:
        n = min(len(left), len(right))
        peak = max_cross_correlation(np.array(left[:n]), np.array(right[:n]))
        assert -1.0 - 1e-9 <= peak <= 1.0 + 1e-9

    @given(st.floats(min_value=0.01, max_value=2.0))
    def test_rmse_scales_with_the_error(self, scale: float) -> None:
        clean = _wave(256)
        error = np.full(clean.size, scale)
        assert rmse(clean, clean + error) == pytest.approx(scale)
