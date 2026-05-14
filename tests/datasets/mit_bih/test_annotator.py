"""Tests for MIT-BIH annotator: clean segment extraction and corruption pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_datasets.mit_bih.annotator import extract_clean_segments, make_pair

# -----------------------------------------------------------------------
# extract_clean_segments
# -----------------------------------------------------------------------


class TestExtractCleanSegments:
    """Tests for artifact-free segment extraction."""

    def test_returns_list_of_arrays(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann: MagicMock,
    ) -> None:
        """Result is a non-empty list of numpy arrays."""
        segments = extract_clean_segments("100", tmp_path, duration_s=5.0)
        assert isinstance(segments, list)
        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg, np.ndarray)

    def test_segment_length_matches_target(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann: MagicMock,
    ) -> None:
        """Each segment has exactly int(duration_s * fs) samples."""
        duration_s = 5.0
        fs = 250.0
        expected_len = int(duration_s * fs)  # 1250

        segments = extract_clean_segments("100", tmp_path, duration_s=duration_s, fs=fs)

        for seg in segments:
            assert seg.shape == (expected_len,), (
                f"Expected length {expected_len}, got {seg.shape}"
            )

    def test_segments_are_normalized(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann: MagicMock,
    ) -> None:
        """Extracted segments have mean ≈ 0 and std ≈ 1."""
        segments = extract_clean_segments("100", tmp_path, duration_s=5.0)

        for seg in segments:
            assert abs(float(np.mean(seg))) < 0.1  # near zero
            assert abs(float(np.std(seg)) - 1.0) < 0.1  # near unit

    def test_artifact_window_excluded(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann: MagicMock,
    ) -> None:
        """Fewer segments returned when an artifact is present.

        The fake record is 60 s with an artifact at 25 s.
        With 10 s windows and no artifact, we'd get 6 segments
        (0-10, 10-20, 20-30, 30-40, 40-50, 50-60).
        With the artifact at 25 s, the 20-30 s window is excluded,
        so we expect fewer segments.
        """
        segments_with_artifact = extract_clean_segments(
            "100", tmp_path, duration_s=10.0
        )
        # 60 s record, 10 s windows → 6 max; artifact at 25 s should
        # eliminate the window containing it, so we get fewer.
        assert len(segments_with_artifact) < 6

    def test_clean_record_gives_more_segments(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """A fully clean record yields more segments than one with artifacts."""
        # mock_rdann_clean has no artifacts → maximum segments
        segments = extract_clean_segments("100", tmp_path, duration_s=10.0)
        # 60 s / 10 s = 6 non-overlapping segments (all clean)
        assert len(segments) == 6

    def test_empty_when_duration_exceeds_record(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Returns empty list when requested duration exceeds record length."""
        # Fake record is 60 s; ask for 120 s
        segments = extract_clean_segments("100", tmp_path, duration_s=120.0)
        assert segments == []

    def test_default_fs_is_250(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Default target fs is 250 Hz (matching Phase 2 SignalGenerator)."""
        segments = extract_clean_segments("100", tmp_path, duration_s=5.0)
        expected_len = int(5.0 * 250.0)  # 1250
        assert segments[0].shape == (expected_len,)

    def test_dtype_is_float64(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Output segments are float64 (from normalize)."""
        segments = extract_clean_segments("100", tmp_path, duration_s=5.0)
        for seg in segments:
            assert seg.dtype == np.float64


# -----------------------------------------------------------------------
# make_pair
# -----------------------------------------------------------------------


class TestMakePair:
    """Tests for the ground-truth corruption pipeline."""

    def test_returns_equal_length_tuple(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """make_pair returns (clean, noisy) with identical shapes."""
        clean, noisy = make_pair(
            "100",
            tmp_path,
            duration_s=5.0,
            flicker_freq=50.0,
            flicker_amplitude=0.5,
            snr_db=10.0,
        )

        assert clean.shape == noisy.shape
        assert clean.ndim == 1

    def test_clean_and_noisy_differ(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Noisy signal differs from the clean reference."""
        clean, noisy = make_pair(
            "100",
            tmp_path,
            duration_s=5.0,
            flicker_freq=50.0,
            flicker_amplitude=0.5,
            snr_db=10.0,
        )

        assert not np.array_equal(clean, noisy)

    def test_snr_approximately_matches_target(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Measured SNR between clean and noisy is close to the target snr_db.

        Uses a looser tolerance because the flicker component adds
        deterministic interference on top of the Gaussian noise, and the
        clean signal is normalized (affecting the overall power budget).
        """
        target_snr = 10.0
        clean, noisy = make_pair(
            "100",
            tmp_path,
            duration_s=10.0,
            flicker_freq=50.0,
            flicker_amplitude=0.0,  # disable flicker to isolate Gaussian SNR
            snr_db=target_snr,
            seed=42,
        )

        measured_snr = calculate_snr(clean, noisy)
        # With only Gaussian noise, measured SNR should be within ±3 dB
        # of the target (statistical variance + normalization effects).
        assert abs(measured_snr - target_snr) < 3.0, (
            f"Measured SNR {measured_snr:.2f} dB too far from target {target_snr} dB"
        )

    def test_deterministic_with_same_seed(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Same seed produces identical (clean, noisy) pairs."""
        kwargs: dict[str, Any] = {
            "record_id": "100",
            "cache_dir": tmp_path,
            "duration_s": 5.0,
            "flicker_freq": 50.0,
            "flicker_amplitude": 0.5,
            "snr_db": 10.0,
            "seed": 123,
        }

        clean1, noisy1 = make_pair(**kwargs)
        clean2, noisy2 = make_pair(**kwargs)

        np.testing.assert_array_equal(clean1, clean2)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_produce_different_noise(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Different seeds produce different noisy signals."""
        kwargs: dict[str, Any] = {
            "record_id": "100",
            "cache_dir": tmp_path,
            "duration_s": 5.0,
            "flicker_freq": 50.0,
            "flicker_amplitude": 0.5,
            "snr_db": 10.0,
        }

        _, noisy1 = make_pair(**kwargs, seed=0)
        _, noisy2 = make_pair(**kwargs, seed=999)

        assert not np.array_equal(noisy1, noisy2)

    def test_no_flicker_when_amplitude_zero(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """When flicker_amplitude=0, only Gaussian noise is applied."""
        clean, noisy_no_flicker = make_pair(
            "100",
            tmp_path,
            duration_s=5.0,
            flicker_freq=50.0,
            flicker_amplitude=0.0,
            snr_db=20.0,
            seed=7,
        )

        clean2, noisy_with_flicker = make_pair(
            "100",
            tmp_path,
            duration_s=5.0,
            flicker_freq=50.0,
            flicker_amplitude=0.5,
            snr_db=20.0,
            seed=7,
        )

        # Both should differ from clean, but in different ways
        noise_only = noisy_no_flicker - clean
        noise_plus_flicker = noisy_with_flicker - clean2

        # The flicker version should have higher energy
        assert np.var(noise_plus_flicker) > np.var(noise_only)

    def test_raises_on_no_clean_segments(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """ValueError raised when duration exceeds available clean data."""
        with pytest.raises(ValueError, match="no artifact-free segment"):
            make_pair(
                "100",
                tmp_path,
                duration_s=120.0,  # exceeds 60 s record
                flicker_freq=50.0,
                flicker_amplitude=0.5,
                snr_db=10.0,
            )

    def test_output_contract_matches_signal_generator(
        self,
        tmp_path: Path,
        mock_rdrecord: MagicMock,
        mock_rdann_clean: MagicMock,
    ) -> None:
        """Return type matches SignalGenerator.generate() contract.

        Both return tuple[np.ndarray, np.ndarray] with 1-D float arrays.
        """
        result = make_pair(
            "100",
            tmp_path,
            duration_s=5.0,
            flicker_freq=50.0,
            flicker_amplitude=0.5,
            snr_db=10.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        clean, noisy = result
        assert isinstance(clean, np.ndarray)
        assert isinstance(noisy, np.ndarray)
        assert clean.ndim == 1
        assert noisy.ndim == 1
        assert clean.dtype == np.float64
