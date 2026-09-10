"""Guards against scoring outputs that carry no signal at all.

A spiking network can fail in two ways that both produce a *constant* output:
every neuron saturates, or every neuron falls silent. Both are total failures,
but scale-invariant metrics do not see them that way. `affine_corrected_snr`
in particular fits the best `a * x + b`; given a constant input it sets `a = 0`
and `b = mean(clean)`, scoring the flat line as though it were a reconstruction.

Phase 4's `0.00 ±0.00` rows are exactly this failure. Without a guard they read
as "a low ceiling"; they actually mean "no output at all". Every reported
metric must be gated on `is_degenerate` returning False.

See `docs/paper_roadmap.md` R1 and Finding 3.
"""

import numpy as np

DEFAULT_MIN_STD = 1e-6


def output_std(signal: np.ndarray) -> float:
    """Standard deviation of a reconstruction, as a degeneracy diagnostic."""
    return float(np.std(np.asarray(signal, dtype=np.float64)))


def is_degenerate(signal: np.ndarray, min_std: float = DEFAULT_MIN_STD) -> bool:
    """
    True if `signal` is constant, empty, or non-finite — i.e. carries no signal.

    Args:
        signal: Reconstruction to check.
        min_std: Standard deviation at or below which the signal counts as
            constant. The default is a numerical-noise floor; experiments
            comparing against a known-good operating point should pass a
            fraction of that point's output standard deviation instead.

    Returns:
        True if the signal cannot meaningfully be scored.

    Raises:
        ValueError: If `min_std` is negative.
    """
    if min_std < 0.0:
        raise ValueError(f"min_std must be non-negative, got {min_std}.")

    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return True
    return bool(np.std(arr) <= min_std)
