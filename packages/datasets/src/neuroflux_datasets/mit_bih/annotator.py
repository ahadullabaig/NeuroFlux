"""Ground-truth annotator and corruption pipeline for MIT-BIH records.

Extracts artifact-free ECG segments from downloaded MIT-BIH records using
``wfdb`` annotations, then constructs ``(clean, noisy)`` signal pairs by
injecting controlled flicker and Gaussian noise — reusing the **identical**
corruption functions from :mod:`neuroflux_datasets.synthetic` so that
Phase 2 and Phase 3 share the same noise math.

The ``wfdb`` import is **lazy** (inside functions that access record files)
so that Phase 2 code paths never depend on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroflux_datasets.mit_bih.preprocessor import normalize, resample_signal
from neuroflux_datasets.synthetic import add_flicker, add_gaussian_noise

# MIT-BIH native sampling rate (all 48 records).
_MITBIH_FS: float = 360.0

# Annotation symbols considered "artifact" or "noise" markers.
# When any of these appear inside a time window, that window is excluded.
#   '~'  — signal quality change / noise
#   '|'  — isolated QRS-like artifact
#   '+'  — rhythm change (may signal pacing / irregular segments)
_ARTIFACT_SYMBOLS: frozenset[str] = frozenset({"~", "|", "+"})


def extract_clean_segments(
    record_id: str,
    cache_dir: str | Path,
    duration_s: float,
    fs: float = 250.0,
    *,
    channel: int = 0,
) -> list[np.ndarray]:
    """Extract artifact-free ECG segments from a cached MIT-BIH record.

    The function reads the record and its beat annotations via ``wfdb``,
    identifies contiguous windows of ``duration_s`` seconds that contain
    **no** artifact/noise annotation symbols, then resamples each window
    from 360 Hz to *fs* and normalizes to zero-mean unit-variance.

    Args:
        record_id: MIT-BIH record identifier (e.g. ``"100"``).
        cache_dir: Directory containing the cached ``.hea`` / ``.dat`` /
            ``.atr`` files (as written by :func:`downloader.download_record`).
        duration_s: Desired segment length in seconds.
        fs: Target sampling frequency after resampling (default 250 Hz,
            matching the Phase 2 ``SignalGenerator``).
        channel: Which signal channel to extract (default ``0`` = MLII).

    Returns:
        List of 1-D ``float64`` arrays, each of length
        ``int(duration_s * fs)`` samples. May be empty if the record
        contains no artifact-free window of the requested duration.
    """
    import wfdb  # type: ignore[import-untyped]  # noqa: PLC0415

    cache_path = Path(cache_dir)
    record_path = str(cache_path / record_id)

    # Read the full record in physical units (mV).
    record = wfdb.rdrecord(record_path, channels=[channel])
    sig: np.ndarray = record.p_signal[:, 0]  # (n_samples,)
    record_fs: float = float(record.fs)

    # Read beat annotations.
    ann = wfdb.rdann(record_path, "atr")
    ann_samples: np.ndarray = np.asarray(ann.sample)
    ann_symbols: list[str] = list(ann.symbol)

    # Number of samples per segment at the *native* rate.
    window_samples = int(duration_s * record_fs)

    if window_samples <= 0 or window_samples > len(sig):
        return []

    # Build a sorted array of artifact sample indices for fast lookup.
    artifact_positions = np.array(
        [
            s
            for s, sym in zip(ann_samples, ann_symbols, strict=True)
            if sym in _ARTIFACT_SYMBOLS
        ],
        dtype=np.int64,
    )

    segments: list[np.ndarray] = []

    # Slide through the record by non-overlapping windows.
    start = 0
    while start + window_samples <= len(sig):
        end = start + window_samples

        # Check for artifact annotations inside [start, end).
        if len(artifact_positions) > 0:
            in_window = (artifact_positions >= start) & (artifact_positions < end)
            if np.any(in_window):
                # Skip to just past the last artifact in this window.
                last_artifact = int(artifact_positions[in_window][-1])
                start = last_artifact + 1
                continue

        # Extract, resample, and normalize the clean window.
        raw_segment = sig[start:end].copy()
        resampled = resample_signal(raw_segment, fs_in=record_fs, fs_out=fs)
        normalized = normalize(resampled)
        segments.append(normalized)

        # Advance past this window (non-overlapping).
        start = end

    return segments


def make_pair(
    record_id: str,
    cache_dir: str | Path,
    duration_s: float,
    flicker_freq: float,
    flicker_amplitude: float,
    snr_db: float,
    fs: float = 250.0,
    seed: int = 0,
    *,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a ``(clean, noisy)`` signal pair from a real ECG segment.

    Extracts a clean segment via :func:`extract_clean_segments`, then
    applies the **same** flicker and noise injection used by the Phase 2
    ``SignalGenerator`` — :func:`synthetic.add_flicker` and
    :func:`synthetic.add_gaussian_noise` — so the corruption math is
    identical across phases.

    The function deterministically selects a segment using *seed* to
    index into the available clean segments (modulo the count), ensuring
    reproducible experiment runs.

    Args:
        record_id: MIT-BIH record identifier.
        cache_dir: Local cache directory with downloaded record files.
        duration_s: Segment duration in seconds.
        flicker_freq: Flicker interference frequency in Hz.
        flicker_amplitude: Flicker interference amplitude.
        snr_db: Target signal-to-noise ratio for Gaussian noise (dB).
        fs: Target sampling frequency (default 250 Hz).
        seed: Random seed for noise generation and segment selection.
        channel: Signal channel index (default ``0`` = MLII).

    Returns:
        Tuple ``(clean, noisy)`` of equal-length 1-D ``float64`` arrays.
        Same contract as ``SignalGenerator.generate()``.

    Raises:
        ValueError: If no artifact-free segment of the requested duration
            is available in the record.
    """
    segments = extract_clean_segments(
        record_id,
        cache_dir,
        duration_s,
        fs,
        channel=channel,
    )

    if not segments:
        msg = (
            f"Record {record_id} has no artifact-free segment of "
            f"{duration_s}s at {fs} Hz."
        )
        raise ValueError(msg)

    # Deterministic segment selection.
    segment_idx = seed % len(segments)
    clean = segments[segment_idx]

    # Apply corruption — identical to SignalGenerator.generate().
    rng = np.random.default_rng(seed)
    noisy = clean.copy()

    if flicker_freq > 0 and flicker_amplitude > 0:
        noisy = add_flicker(noisy, fs, flicker_freq, flicker_amplitude)

    noisy = add_gaussian_noise(noisy, snr_db, rng)

    return clean, noisy
