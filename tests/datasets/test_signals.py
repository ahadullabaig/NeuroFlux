"""Tests for the unified experiment signal source.

The MIT-BIH tests skip cleanly when records are not cached, matching
`test_phase3_regression.py`, so CI without bundled data still passes.
"""

from pathlib import Path

import numpy as np
import pytest
from neuroflux_datasets.signals import (
    DEFAULT_CACHE_DIR,
    PHASE3_RECORDS,
    load_clean_ecg,
)

_CACHED = Path(DEFAULT_CACHE_DIR) / "214.hea"
requires_records = pytest.mark.skipif(
    not _CACHED.exists(), reason="MIT-BIH record 214 not cached"
)


def test_synthetic_has_the_requested_length() -> None:
    signal = load_clean_ecg("synthetic", duration_s=4.0, fs=250.0)
    assert signal.shape == (1000,)
    assert np.all(np.isfinite(signal))


def test_synthetic_respects_sampling_rate() -> None:
    assert load_clean_ecg("synthetic", duration_s=2.0, fs=360.0).shape == (720,)


def test_synthetic_ignores_unit_and_window() -> None:
    """Generated ECG is deterministic, so these knobs are real-data only."""
    a = load_clean_ecg("synthetic", unit="214", window=3)
    b = load_clean_ecg("synthetic", unit="999", window=7)
    assert np.array_equal(a, b)


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_clean_ecg("eeg-lab")


def test_uncached_record_names_the_fix() -> None:
    with pytest.raises(FileNotFoundError, match="download_record"):
        load_clean_ecg("mit-bih", unit="100", cache_dir="/nonexistent/path")


def test_phase3_records_are_the_checksummed_sixteen() -> None:
    assert len(PHASE3_RECORDS) == 16
    assert len(set(PHASE3_RECORDS)) == 16
    assert "214" in PHASE3_RECORDS


@requires_records
def test_real_record_has_the_requested_length() -> None:
    signal = load_clean_ecg("mit-bih", unit="214", duration_s=8.0, fs=250.0)
    assert signal.shape == (2000,)
    assert np.all(np.isfinite(signal))


@requires_records
def test_real_record_is_normalised() -> None:
    """`extract_clean_segments` normalises, so downstream scaling is uniform."""
    signal = load_clean_ecg("mit-bih", unit="214", duration_s=8.0)
    assert abs(float(np.mean(signal))) < 0.1
    assert abs(float(np.std(signal)) - 1.0) < 0.1


@requires_records
def test_window_selects_different_segments() -> None:
    """A seed index passed as `window` must vary the signal, not repeat it."""
    first = load_clean_ecg("mit-bih", unit="214", duration_s=8.0, window=0)
    second = load_clean_ecg("mit-bih", unit="214", duration_s=8.0, window=1)
    assert not np.array_equal(first, second)


@requires_records
def test_window_wraps_rather_than_raising() -> None:
    signal = load_clean_ecg("mit-bih", unit="214", duration_s=8.0, window=10_000)
    assert signal.shape == (2000,)


@requires_records
def test_impossible_window_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="no artifact-free"):
        load_clean_ecg("mit-bih", unit="214", duration_s=100_000.0)
