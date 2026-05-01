"""Classification accuracy and decision utilities."""

import numpy as np


def winner_take_all(spike_counts: np.ndarray) -> int:
    """Return the index of the highest-firing neuron (ties broken by argmax)."""
    arr = np.asarray(spike_counts)
    if arr.ndim != 1:
        raise ValueError("spike_counts must be 1D")
    if arr.size == 0:
        raise ValueError("spike_counts must be non-empty")
    return int(np.argmax(arr))


def classification_accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
    """Fraction of equal entries in `predicted` and `true`."""
    p = np.asarray(predicted)
    t = np.asarray(true)
    if p.shape != t.shape:
        raise ValueError("predicted and true must have the same shape")
    if p.size == 0:
        raise ValueError("inputs must be non-empty")
    return float(np.mean(p == t))
