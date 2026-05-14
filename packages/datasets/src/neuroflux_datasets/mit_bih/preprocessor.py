"""Signal preprocessing for MIT-BIH Arrhythmia Database records.

Provides resampling (360 → 250 Hz) and per-channel normalization to prepare
raw wfdb records for the NeuroFlux SNN pipeline. The 250 Hz target matches
the Phase 2 ``SignalGenerator`` default so that downstream network
configurations transfer directly.
"""

import math

import numpy as np
from scipy.signal import resample_poly  # type: ignore[import-untyped]


def resample_signal(
    sig: np.ndarray,
    fs_in: float = 360.0,
    fs_out: float = 250.0,
) -> np.ndarray:
    """Resample a signal from *fs_in* to *fs_out* Hz.

    Uses polyphase anti-aliased resampling (``scipy.signal.resample_poly``).
    The up/down factors are derived from the integer ratio ``fs_out / fs_in``
    reduced by their GCD (e.g. 360 → 250 becomes up=25, down=36).

    Args:
        sig: Input signal. Shape ``(n_samples,)`` for single-channel or
            ``(n_samples, n_channels)`` for multi-channel.
        fs_in: Input sampling frequency in Hz (must be a positive integer
            value, though typed as float for convenience).
        fs_out: Output sampling frequency in Hz (same constraint).

    Returns:
        Resampled signal with approximately
        ``n_samples * fs_out / fs_in`` samples along the time axis.
        Channel count (if 2-D) is preserved.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(
            f"Sampling frequencies must be positive, got fs_in={fs_in}, fs_out={fs_out}"
        )

    # No-op fast path: same rate → return a copy (avoids filter artifacts)
    if fs_in == fs_out:
        return np.array(sig, dtype=sig.dtype)

    up = int(fs_out)
    down = int(fs_in)
    g = math.gcd(up, down)
    up //= g
    down //= g

    resampled: np.ndarray = resample_poly(sig, up, down, axis=0)
    return resampled


def normalize(sig: np.ndarray) -> np.ndarray:
    """Normalize a signal to zero-mean, unit-variance per channel.

    For 2-D inputs each column (channel) is normalized independently.
    Channels with zero variance (constant value) are set to all-zeros
    rather than producing ``NaN`` / ``Inf`` from division.

    Args:
        sig: Input signal. Shape ``(n_samples,)`` or
            ``(n_samples, n_channels)``.

    Returns:
        Normalized copy of the signal as ``float64``.
    """
    result = np.array(sig, dtype=np.float64)

    if result.ndim == 1:
        mean = float(np.mean(result))
        std = float(np.std(result))
        if std == 0.0:
            return np.zeros_like(result)
        return (result - mean) / std

    # 2-D: vectorized per-column normalization
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0)

    # Guard zero-variance channels: use 1.0 as divisor to avoid div-by-zero,
    # then overwrite those columns with zeros afterward.
    zero_var_mask: np.ndarray = stds == 0.0
    safe_stds: np.ndarray = np.where(zero_var_mask, 1.0, stds)

    result = np.asarray((result - means) / safe_stds)
    result[:, zero_var_mask] = 0.0

    return result
