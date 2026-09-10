"""Reconstruction-quality metrics for signal-denoising evaluation.

`calculate_snr` compares a reconstruction against the clean signal on an
absolute scale, which penalises any constant offset or gain mismatch. A
spiking pipeline's output lives on whatever scale the decoder happens to
produce, so an otherwise faithful reconstruction can score arbitrarily badly
for reasons that have nothing to do with denoising.

`affine_corrected_snr` removes that confound by fitting the least-squares
`a * x + b` that best maps the reconstruction onto the clean signal before
scoring. What remains is the part of the error that no gain-and-offset
adjustment can fix — the shape error, which is what we actually mean by
reconstruction quality.

Always pair this with `metrics.degeneracy.is_degenerate`: a constant output
has no shape to get wrong, so the affine fit collapses it onto the clean
signal's mean and the score becomes meaningless rather than bad.
"""

import numpy as np

from neuroflux_core.metrics.snr import calculate_snr


def affine_fit(signal_true: np.ndarray, signal_pred: np.ndarray) -> tuple[float, float]:
    """
    Least-squares scale and offset mapping `signal_pred` onto `signal_true`.

    Args:
        signal_true: Reference signal.
        signal_pred: Signal to be rescaled.

    Returns:
        `(a, b)` minimising `||signal_true - (a * signal_pred + b)||^2`.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    true_f = np.asarray(signal_true, dtype=np.float64).ravel()
    pred_f = np.asarray(signal_pred, dtype=np.float64).ravel()

    if true_f.size == 0 or pred_f.size == 0:
        raise ValueError("Signals must be non-empty.")
    if true_f.size != pred_f.size:
        raise ValueError(
            f"Length mismatch: {true_f.size} true vs {pred_f.size} predicted."
        )

    design = np.vstack([pred_f, np.ones_like(pred_f)]).T
    solution, *_ = np.linalg.lstsq(design, true_f, rcond=None)
    return float(solution[0]), float(solution[1])


def affine_corrected_snr(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """
    SNR in dB after removing the best global scale and offset.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstruction to score.

    Returns:
        SNR in dB of the affine-corrected reconstruction.

    Raises:
        ValueError: If the inputs differ in length or are empty.

    Note:
        A degenerate (near-constant) `signal_pred` yields a meaningless score.
        Guard with `metrics.degeneracy.is_degenerate` before interpreting.
    """
    scale, offset = affine_fit(signal_true, signal_pred)
    pred_f = np.asarray(signal_pred, dtype=np.float64).ravel()
    return calculate_snr(signal_true, scale * pred_f + offset)
