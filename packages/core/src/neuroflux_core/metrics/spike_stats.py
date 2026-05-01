"""Spike-train statistics: firing rates, silent neurons, Fano factor."""

import numpy as np


def mean_firing_rate(spike_train: np.ndarray, dt: float) -> np.ndarray:
    """Per-neuron mean firing rate (Hz) over a (T, N) spike train."""
    arr = np.asarray(spike_train, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("spike_train must be 2D (T, N)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return np.asarray(arr.mean(axis=0) * (1000.0 / dt), dtype=np.float64)


def silent_neuron_fraction(spike_train: np.ndarray) -> float:
    """Fraction of neurons that never fired across the (T, N) spike train."""
    arr = np.asarray(spike_train)
    if arr.ndim != 2:
        raise ValueError("spike_train must be 2D (T, N)")
    if arr.shape[1] == 0:
        return 0.0
    return float(np.mean(arr.sum(axis=0) == 0))


def fano_factor(spike_train: np.ndarray, dt: float, window_ms: float) -> np.ndarray:
    """
    Fano factor (variance / mean of spike counts in `window_ms` windows) per
    neuron. Returns 0.0 for neurons with zero mean count.
    """
    arr = np.asarray(spike_train, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("spike_train must be 2D (T, N)")
    if dt <= 0.0 or window_ms <= 0.0:
        raise ValueError("dt and window_ms must be positive")

    win = max(1, int(round(window_ms / dt)))
    n_windows = arr.shape[0] // win
    n_neurons = arr.shape[1]

    if n_windows == 0:
        return np.zeros(n_neurons, dtype=np.float64)

    counts = arr[: n_windows * win].reshape(n_windows, win, n_neurons).sum(axis=1)
    mean = counts.mean(axis=0)
    var = counts.var(axis=0)
    out = np.zeros(n_neurons, dtype=np.float64)
    nonzero = mean > 0
    out[nonzero] = var[nonzero] / mean[nonzero]
    return out
