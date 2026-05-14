"""R-peak integrity metric for ECG signal evaluation.

Measures how well an SNN-reconstructed ECG preserves the clinically
critical R-peak locations of the original signal.  R-peaks are detected
via ``scipy.signal.find_peaks`` and matched between the clean reference
and the reconstruction within a configurable time tolerance.
"""

import numpy as np
from scipy.signal import find_peaks  # type: ignore[import-untyped]


def rpeak_integrity(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    fs: float,
    tol_ms: float = 50.0,
) -> float:
    """Compute the R-peak integrity score between two ECG signals.

    Detects R-peaks in both the *clean* reference and the
    *reconstructed* signal, then matches them within a time tolerance.
    The score is the fraction of clean R-peaks that have a matching
    reconstructed R-peak.

    Args:
        clean: Ground-truth 1-D ECG signal.
        reconstructed: SNN-reconstructed 1-D ECG signal (same length).
        fs: Sampling frequency in Hz.
        tol_ms: Maximum allowed temporal shift (in milliseconds) for a
            reconstructed R-peak to be considered a match.  Default
            50 ms follows clinical QRS-detection tolerance conventions.

    Returns:
        Score in ``[0.0, 1.0]``.  ``1.0`` means every clean R-peak was
        recovered; ``0.0`` means none were (or there are no R-peaks in
        the clean signal).

    Raises:
        ValueError: If *clean* and *reconstructed* have different shapes.
    """
    if clean.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: clean {clean.shape} vs "
            f"reconstructed {reconstructed.shape}."
        )

    # Tolerance in samples.
    tol_samples = int(round(tol_ms * fs / 1000.0))

    # Detect R-peaks using prominence-based detection.
    # Use the signal's standard deviation as the minimum prominence
    # threshold to filter out low-amplitude noise peaks.
    clean_std = float(np.std(clean))
    recon_std = float(np.std(reconstructed))

    # Minimum prominence: 0.5 × std (tuned for normalized ECG signals).
    clean_prom = max(clean_std * 0.5, 1e-10)
    recon_prom = max(recon_std * 0.5, 1e-10)

    clean_peaks, _ = find_peaks(clean, prominence=clean_prom)
    recon_peaks, _ = find_peaks(reconstructed, prominence=recon_prom)

    # No R-peaks in clean signal → score is 0.0 (nothing to match).
    if len(clean_peaks) == 0:
        return 0.0

    # No R-peaks in reconstruction → none matched.
    if len(recon_peaks) == 0:
        return 0.0

    # Greedy matching: for each clean R-peak, find the closest
    # reconstructed R-peak within tolerance.
    recon_peaks_sorted = np.sort(recon_peaks)
    matched = 0

    # Track which recon peaks have been used (prevent double-matching).
    used: set[int] = set()

    for cp in clean_peaks:
        # Binary search for the closest reconstructed peak.
        idx = int(np.searchsorted(recon_peaks_sorted, cp))

        # Check the candidate and its neighbour (searchsorted gives
        # the insertion point, so the closest could be idx or idx-1).
        best_dist = tol_samples + 1
        best_idx = -1

        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(recon_peaks_sorted):
                if candidate_idx not in used:
                    dist = abs(int(recon_peaks_sorted[candidate_idx]) - int(cp))
                    if dist <= tol_samples and dist < best_dist:
                        best_dist = dist
                        best_idx = candidate_idx

        if best_idx >= 0:
            matched += 1
            used.add(best_idx)

    return float(matched / len(clean_peaks))
