"""Classical baseline filters for ECG denoising comparison."""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch  # type: ignore[import-untyped]


def butterworth_notch(
    signal: np.ndarray,
    freq_hz: float,
    fs: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply an IIR notch filter centered on ``freq_hz``.

    Uses ``scipy.signal.iirnotch`` to design the filter and ``filtfilt`` for
    zero-phase application. This is the canonical clinical denoiser for
    powerline interference at a known frequency.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"butterworth_notch expects a 1-D signal, got shape {signal.shape}"
        )
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = fs / 2.0
    if not 0.0 < freq_hz < nyquist:
        raise ValueError(f"freq_hz={freq_hz} must lie in (0, {nyquist}) for fs={fs}")

    b, a = iirnotch(freq_hz, quality_factor, fs)
    return np.asarray(filtfilt(b, a, signal), dtype=signal.dtype)


def lms(
    signal: np.ndarray,
    mu: float = 0.5,
    filter_order: int = 32,
    eps: float = 1e-6,
) -> np.ndarray:
    """Self-referencing normalized LMS (NLMS) adaptive filter.

    Predicts each sample from the preceding ``filter_order`` samples and
    returns the prediction error as the denoised output. Uses NLMS rather
    than plain LMS: the per-step update is ``mu * e * x / (x·x + eps)``,
    which makes the step size auto-adaptive to local signal power. NLMS is
    stable for ``0 < mu < 2`` regardless of input amplitude — plain LMS
    diverges on real ECG when the signal variance differs from the synthetic
    test cases the step size was tuned for.

    Apples-to-apples with NeuroFlux: unsupervised, single-channel, no
    reference signal.
    """
    if signal.ndim != 1:
        raise ValueError(f"lms expects a 1-D signal, got shape {signal.shape}")
    if filter_order < 1:
        raise ValueError(f"filter_order must be >= 1, got {filter_order}")
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    n = len(signal)
    w = np.zeros(filter_order, dtype=np.float64)
    output = np.zeros(n, dtype=np.float64)
    sig = signal.astype(np.float64, copy=False)
    for i in range(filter_order, n):
        x = sig[i - filter_order : i][::-1]
        y_hat = float(np.dot(w, x))
        e = sig[i] - y_hat
        norm = float(np.dot(x, x)) + eps
        w += mu * e * x / norm
        output[i] = e
    return output.astype(signal.dtype, copy=False)
