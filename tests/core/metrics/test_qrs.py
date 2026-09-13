"""Tests for annotation-referenced QRS scoring.

The load-bearing test in this module is `TestSaturation`: it pins the defect
that motivated the metric. `rpeak_integrity` returns 1.000 for the unprocessed
noisy signal, so it certifies doing nothing as perfect preservation; the
replacement must not. Everything else here is the matching arithmetic that
makes that comparison trustworthy.
"""

import math
import sys

import numpy as np
import pytest
from neuroflux_core.metrics.qrs import (
    DEFAULT_GUARD_MS,
    DEFAULT_TOLERANCE_MS,
    QrsScore,
    detect_rpeaks,
    match_beats,
    qrs_score,
)
from neuroflux_core.metrics.rpeak import rpeak_integrity

FS = 250.0

# 150 ms at 250 Hz.
TOL_SAMPLES = 37


def _beats(*positions: int) -> np.ndarray:
    """Beat positions as the array shape the metric expects."""
    return np.array(positions, dtype=np.int64)


class TestPerfectAgreement:
    """A detector that reproduces the annotation exactly."""

    def test_every_field_is_perfect(self) -> None:
        beats = _beats(100, 350, 600, 850)
        score = match_beats(beats, beats, FS)
        assert score == QrsScore(
            n_reference=4,
            n_detected=4,
            true_positives=4,
            false_positives=0,
            false_negatives=0,
            sensitivity=1.0,
            ppv=1.0,
            f1=1.0,
            rms_jitter_ms=0.0,
            mean_offset_ms=0.0,
        )

    def test_a_beat_just_inside_tolerance_still_matches(self) -> None:
        reference = _beats(500)
        score = match_beats(reference, _beats(500 + TOL_SAMPLES - 1), FS)
        assert score.true_positives == 1

    def test_a_beat_outside_tolerance_does_not_match(self) -> None:
        reference = _beats(500)
        score = match_beats(reference, _beats(500 + TOL_SAMPLES + 2), FS)
        assert score.true_positives == 0
        assert score.false_positives == 1
        assert score.false_negatives == 1


class TestOneToOneMatching:
    """Over-detection must cost PPV, not be absorbed silently."""

    def test_two_detections_on_one_beat_give_one_false_positive(self) -> None:
        score = match_beats(_beats(500), _beats(495, 510), FS)
        assert (score.true_positives, score.false_positives) == (1, 1)
        assert score.sensitivity == 1.0
        assert score.ppv == pytest.approx(0.5)

    def test_the_closer_of_two_detections_is_the_one_that_matches(self) -> None:
        # A left-to-right pass would take 470 and leave 505 unmatched, which
        # would report a 120 ms error instead of a 20 ms one.
        score = match_beats(_beats(500), _beats(470, 505), FS)
        assert score.rms_jitter_ms == pytest.approx(20.0)

    def test_a_detection_is_not_reused_across_two_reference_beats(self) -> None:
        score = match_beats(_beats(500, 520), _beats(510), FS)
        assert (score.true_positives, score.false_negatives) == (1, 1)

    def test_proximity_order_wins_over_position_order(self) -> None:
        # Detection 505 is nearer reference 500 than reference 520 is; a
        # position-ordered pass would pair (500, 525) and (520, 505), inverting
        # the assignment and inflating the jitter.
        score = match_beats(_beats(500, 520), _beats(505, 525), FS)
        assert score.true_positives == 2
        assert score.rms_jitter_ms == pytest.approx(20.0)


class TestEmptyInputs:
    """Total detector failure scores zero rather than raising."""

    def test_no_detections_scores_zero_with_nan_jitter(self) -> None:
        score = match_beats(_beats(100, 350), _beats(), FS)
        assert (score.sensitivity, score.ppv, score.f1) == (0.0, 0.0, 0.0)
        assert math.isnan(score.rms_jitter_ms)
        assert math.isnan(score.mean_offset_ms)

    def test_no_reference_beats_scores_zero(self) -> None:
        score = match_beats(_beats(), _beats(100, 350), FS)
        assert score.false_positives == 2
        assert score.f1 == 0.0

    def test_both_empty_scores_zero(self) -> None:
        score = match_beats(_beats(), _beats(), FS)
        assert score.f1 == 0.0
        assert score.n_reference == 0


class TestTimingError:
    """Jitter and offset answer different questions."""

    def test_rms_jitter_is_in_milliseconds(self) -> None:
        # Five samples at 250 Hz is 20 ms.
        score = match_beats(_beats(500), _beats(505), FS)
        assert score.rms_jitter_ms == pytest.approx(20.0)

    def test_a_systematic_delay_shows_up_as_a_signed_offset(self) -> None:
        reference = _beats(200, 500, 800)
        score = match_beats(reference, reference + 5, FS)
        assert score.mean_offset_ms == pytest.approx(20.0)
        assert score.rms_jitter_ms == pytest.approx(20.0)

    def test_symmetric_errors_cancel_in_the_offset_but_not_the_jitter(self) -> None:
        score = match_beats(_beats(200, 800), _beats(205, 795), FS)
        assert score.mean_offset_ms == pytest.approx(0.0)
        assert score.rms_jitter_ms == pytest.approx(20.0)


class TestEdgeGuard:
    """A detector's warm-up is not a missed beat."""

    def test_the_guard_drops_beats_near_both_edges(self) -> None:
        reference = _beats(10, 500, 990)
        score = match_beats(reference, reference, FS, guard_ms=100.0, n_samples=1000)
        assert score.n_reference == 1

    def test_without_the_guard_every_beat_is_scored(self) -> None:
        reference = _beats(10, 500, 990)
        assert match_beats(reference, reference, FS).n_reference == 3

    def test_the_guard_applies_to_detections_too(self) -> None:
        # A spurious detection in the guard band is not a false positive.
        score = match_beats(
            _beats(500), _beats(5, 500), FS, guard_ms=100.0, n_samples=1000
        )
        assert score.false_positives == 0
        assert score.ppv == 1.0

    def test_the_guard_needs_the_window_length(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            match_beats(_beats(500), _beats(500), FS, guard_ms=100.0)

    def test_a_negative_guard_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            match_beats(_beats(500), _beats(500), FS, guard_ms=-1.0, n_samples=10)

    def test_the_default_guard_is_one_beat_interval_at_sixty_bpm(self) -> None:
        assert DEFAULT_GUARD_MS == 1000.0


class TestValidation:
    """Bad arguments fail loudly."""

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="fs must be positive"):
            match_beats(_beats(1), _beats(1), 0.0)

    def test_non_positive_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tol_ms must be positive"):
            match_beats(_beats(1), _beats(1), FS, tol_ms=0.0)

    def test_two_dimensional_beats_raise(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            match_beats(np.zeros((2, 2), dtype=np.int64), _beats(1), FS)

    def test_the_default_tolerance_follows_ec57(self) -> None:
        assert DEFAULT_TOLERANCE_MS == 150.0


class TestDetection:
    """The detector wrapper, including its refusal to invent beats."""

    def test_a_flat_signal_yields_no_detections(self) -> None:
        assert detect_rpeaks(np.zeros(2500), FS).size == 0

    def test_a_non_finite_signal_yields_no_detections(self) -> None:
        signal = np.zeros(2500)
        signal[10] = np.nan
        assert detect_rpeaks(signal, FS).size == 0

    def test_an_empty_signal_yields_no_detections(self) -> None:
        assert detect_rpeaks(np.array([]), FS).size == 0

    def test_a_two_dimensional_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            detect_rpeaks(np.zeros((2, 10)), FS)

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="fs must be positive"):
            detect_rpeaks(np.zeros(10), 0.0)

    def test_a_missing_neurokit_names_the_fix(self, monkeypatch: object) -> None:
        # Binding a module to None in sys.modules makes `import` raise.
        monkeypatch.setitem(sys.modules, "neurokit2", None)  # type: ignore[attr-defined]
        with pytest.raises(ImportError, match="uv add neurokit2"):
            detect_rpeaks(np.linspace(0.0, 1.0, 2500), FS)

    def test_it_finds_the_beats_in_a_simulated_ecg(self) -> None:
        nk = pytest.importorskip("neurokit2")
        ecg = nk.ecg_simulate(duration=10, sampling_rate=int(FS), random_state=0)
        peaks = detect_rpeaks(np.asarray(ecg), FS)
        # Ten seconds of simulated ECG at the default 70 bpm.
        assert 8 <= peaks.size <= 14
        assert np.all(np.diff(peaks) > 0)


class TestSaturation:
    """The R1 acceptance criterion: the metric must discriminate.

    `rpeak_integrity` sets each signal's peak prominence floor to a fraction of
    that signal's own standard deviation, so the threshold tracks the noise it
    should penalise. These tests pin both halves: that the old metric certifies
    the identity baseline as perfect, and that the new one does not.
    """

    @staticmethod
    def _simulated_pair(snr_db: float) -> tuple[np.ndarray, np.ndarray]:
        """A simulated ECG and a noisy copy at the requested SNR."""
        nk = pytest.importorskip("neurokit2")
        clean = np.asarray(
            nk.ecg_simulate(duration=20, sampling_rate=int(FS), random_state=0)
        )
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(clean.size)
        scale = math.sqrt(
            float(np.var(clean)) / 10.0 ** (snr_db / 10.0) / float(np.var(noise))
        )
        return clean, clean + scale * noise

    @pytest.mark.parametrize("snr_db", [-10.0, -5.0, 0.0])
    def test_the_old_metric_scores_the_identity_baseline_perfectly(
        self, snr_db: float
    ) -> None:
        clean, noisy = self._simulated_pair(snr_db)
        assert rpeak_integrity(clean, noisy, FS) == pytest.approx(1.0)

    @pytest.mark.parametrize("snr_db", [-10.0, -5.0, 0.0])
    def test_the_new_metric_does_not(self, snr_db: float) -> None:
        clean, noisy = self._simulated_pair(snr_db)
        reference = detect_rpeaks(clean, FS)
        assert qrs_score(reference, noisy, FS).f1 < 0.9

    def test_the_new_metric_still_scores_a_clean_signal_perfectly(self) -> None:
        clean, _ = self._simulated_pair(0.0)
        reference = detect_rpeaks(clean, FS)
        assert qrs_score(reference, clean, FS).f1 == pytest.approx(1.0)

    def test_the_new_metric_is_monotone_in_input_snr(self) -> None:
        clean, _ = self._simulated_pair(0.0)
        reference = detect_rpeaks(clean, FS)
        scores = [
            qrs_score(reference, self._simulated_pair(snr_db)[1], FS).f1
            for snr_db in (-10.0, 0.0, 10.0, 20.0)
        ]
        assert scores == sorted(scores)
        assert scores[-1] - scores[0] > 0.3


class TestEndToEnd:
    """`qrs_score` wires detection and matching together."""

    def test_it_matches_the_two_step_form(self) -> None:
        nk = pytest.importorskip("neurokit2")
        ecg = np.asarray(
            nk.ecg_simulate(duration=10, sampling_rate=int(FS), random_state=1)
        )
        reference = detect_rpeaks(ecg, FS)
        expected = match_beats(
            reference,
            detect_rpeaks(ecg, FS),
            FS,
            guard_ms=DEFAULT_GUARD_MS,
            n_samples=ecg.size,
        )
        assert qrs_score(reference, ecg, FS) == expected

    def test_a_collapsed_output_scores_zero_rather_than_raising(self) -> None:
        score = qrs_score(_beats(500, 1000), np.zeros(2500), FS)
        assert score.f1 == 0.0
        assert score.n_detected == 0
