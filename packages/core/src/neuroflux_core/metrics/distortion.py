"""Waveform-distortion metrics in the vocabulary the ECG literature uses.

SNR in dB is the right primary metric for this work, but it is not what an ECG
reader recognises. Compression and denoising papers report **PRD** — percentage
root-mean-square difference — and a reader who wants to place NeuroFlux next to
a wavelet denoiser needs that number, not a conversion rule.

Two variants are in circulation and they are not interchangeable:

    PRD  = 100 * sqrt( sum (x - xhat)^2 / sum x^2 )
    PRD1 = 100 * sqrt( sum (x - xhat)^2 / sum (x - mean x)^2 )

`prd` divides by the raw energy, so a record with a DC offset inflates the
denominator and the score improves for a reason that has nothing to do with
reconstruction. `prd1` removes the mean first and is therefore comparable
across records; it is the variant to report as primary, and the one Zigel et
al. (2000) argued for on exactly these grounds. Both are here because the
literature quotes both and rarely says which.

`prd1` is the mean-inclusive counterpart of `calculate_snr`: where the SNR
takes the *variance* of the residual, PRD1 takes its raw second moment, so the
two agree only when the residual has zero mean. A systematic offset shows up in
PRD1 and is invisible to the SNR — which is the point of reporting both.

`max_cross_correlation` scores morphology under a timing tolerance. A decoder
with a fixed group delay reconstructs the waveform faithfully and still scores
badly on every sample-aligned metric; the cross-correlation peak separates
"wrong shape" from "right shape, shifted", and the lag says by how much.
"""

import numpy as np
from scipy import signal  # type: ignore[import-untyped]


def _as_pair(
    signal_true: np.ndarray, signal_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten a (reference, estimate) pair to float64."""
    true_f = np.asarray(signal_true, dtype=np.float64).ravel()
    pred_f = np.asarray(signal_pred, dtype=np.float64).ravel()

    if true_f.size == 0 or pred_f.size == 0:
        raise ValueError("Signals must be non-empty.")
    if true_f.size != pred_f.size:
        raise ValueError(
            f"Length mismatch: {true_f.size} true vs {pred_f.size} predicted."
        )
    return true_f, pred_f


def _percentage(residual_energy: float, reference_energy: float) -> float:
    """Shared tail of `prd` and `prd1`, including the zero-energy cases."""
    if reference_energy == 0.0:
        return 0.0 if residual_energy == 0.0 else float("inf")
    return float(100.0 * np.sqrt(residual_energy / reference_energy))


def prd(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """
    Percentage root-mean-square difference, referenced to raw signal energy.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.

    Returns:
        PRD as a percentage; 0.0 for an exact match, `inf` when the reference
        carries no energy but the residual does.

    Raises:
        ValueError: If the inputs differ in length or are empty.

    Note:
        Sensitive to a DC offset in `signal_true`. Prefer `prd1` when comparing
        across records. See the module docstring.
    """
    true_f, pred_f = _as_pair(signal_true, signal_pred)
    residual = true_f - pred_f
    return _percentage(float(residual @ residual), float(true_f @ true_f))


def prd1(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """
    Percentage root-mean-square difference, referenced to signal energy about
    its mean.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.

    Returns:
        PRD1 as a percentage; 0.0 for an exact match, `inf` when the reference
        is constant but the residual is not.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    true_f, pred_f = _as_pair(signal_true, signal_pred)
    residual = true_f - pred_f
    centred = true_f - float(np.mean(true_f))
    return _percentage(float(residual @ residual), float(centred @ centred))


def rmse(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """
    Root-mean-square error, in the units of the input signals.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.

    Returns:
        RMSE, on the same scale as the inputs.

    Raises:
        ValueError: If the inputs differ in length or are empty.

    Note:
        Scale-dependent, so it is only comparable between signals normalised
        the same way. Phase 3 normalises every record to unit variance, which
        makes it comparable there.
    """
    true_f, pred_f = _as_pair(signal_true, signal_pred)
    residual = true_f - pred_f
    return float(np.sqrt(float(residual @ residual) / residual.size))


def cross_correlation_peak(
    signal_true: np.ndarray,
    signal_pred: np.ndarray,
    max_lag_samples: int | None = None,
) -> tuple[float, int]:
    """
    Largest normalised cross-correlation between two signals, and its lag.

    Both signals are mean-centred, correlated at every lag, and normalised by
    the product of their full-signal L2 norms — the convention the ECG
    literature uses. Normalising globally rather than per-lag costs a little
    peak height at large lags, where the sequences overlap less; over the lags
    that matter for a decoder's group delay (tens of milliseconds against a
    multi-second window) the loss is well under a percent.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.
        max_lag_samples: Largest absolute lag to search, in samples. `None`
            searches every lag.

    Returns:
        `(peak, lag)` where `peak` is in `[-1, 1]` and `lag` is the shift in
        samples applied to `signal_pred`: a positive lag means the
        reconstruction is *late* relative to the reference. `(0.0, 0)` when
        either signal is constant, since correlation is undefined there.

    Raises:
        ValueError: If the inputs differ in length, are empty, or if
            `max_lag_samples` is negative.
    """
    true_f, pred_f = _as_pair(signal_true, signal_pred)
    n = true_f.size

    if max_lag_samples is not None and max_lag_samples < 0:
        raise ValueError(f"max_lag_samples must be non-negative, got {max_lag_samples}")

    a = true_f - true_f.mean()
    b = pred_f - pred_f.mean()
    denom = float(np.sqrt(float(a @ a) * float(b @ b)))
    if denom == 0.0:
        return 0.0, 0

    # correlate(a, b) index k holds sum_i a[i] * b[i - (k - (n - 1))], so the
    # lag of index k is (n - 1) - k: the shift applied to `b` = the prediction.
    raw = signal.correlate(a, b, mode="full", method="auto")
    lags = (n - 1) - np.arange(raw.size)

    if max_lag_samples is not None:
        keep = np.abs(lags) <= max_lag_samples
        raw = raw[keep]
        lags = lags[keep]

    best = int(np.argmax(raw))
    return float(raw[best] / denom), int(lags[best])


def max_cross_correlation(
    signal_true: np.ndarray,
    signal_pred: np.ndarray,
    max_lag_samples: int | None = None,
) -> float:
    """
    Largest normalised cross-correlation between two signals.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.
        max_lag_samples: Largest absolute lag to search, in samples. `None`
            searches every lag with at least two overlapping samples.

    Returns:
        The correlation peak in `[-1, 1]`; 1.0 for a perfect match at some lag.

    Raises:
        ValueError: If the inputs differ in length, are empty, or if
            `max_lag_samples` is negative.

    Note:
        A high peak at a large lag means the morphology survived but the timing
        did not. Use `cross_correlation_peak` when that distinction matters.
    """
    peak, _ = cross_correlation_peak(signal_true, signal_pred, max_lag_samples)
    return peak
