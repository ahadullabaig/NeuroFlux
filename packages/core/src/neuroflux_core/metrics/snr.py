"""Signal-to-Noise Ratio (SNR) calculations."""

import numpy as np


def calculate_snr(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels (dB).

    SNR is defined as 10 * log10( variance(signal) / variance(noise) ),
    where noise = signal_true - signal_pred.

    Args:
        signal_true: Ground truth analog signal array.
        signal_pred: Reconstructed analog signal array from SNN output.

    Returns:
        The SNR in dB. If noise variance is zero, returns float('inf').
    """
    if signal_true.shape != signal_pred.shape:
        raise ValueError("signal_true and signal_pred must have the same shape.")

    var_signal = np.var(signal_true)
    noise = signal_true - signal_pred
    var_noise = np.var(noise)

    if var_noise == 0.0:
        return float("inf")

    return float(10.0 * np.log10(var_signal / var_noise))
