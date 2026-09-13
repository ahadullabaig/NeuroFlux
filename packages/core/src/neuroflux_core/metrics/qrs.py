"""QRS detection scored against annotations, replacing `rpeak_integrity`.

`metrics.rpeak.rpeak_integrity` is saturated and must not be used for new
results. It detects peaks in the clean reference *and* in the reconstruction
with `scipy.signal.find_peaks`, setting each prominence floor to `0.5 * std` of
**that signal alone** (`rpeak.py:54-62`). The threshold therefore rises with the
noise it is supposed to be penalised for, and the metric returns 1.000 for the
unprocessed noisy signal all the way down to -20 dB input SNR with 6x flicker.
A metric that scores the identity baseline perfectly cannot separate anything.

Two things are wrong and both are fixed here.

**The reference is circular.** Scoring detections against detections asks
whether two runs of the same detector agree, not whether the beats are right.
The ground truth for a MIT-BIH record is the cardiologist annotation in the
`.atr` file, and that is what `match_beats` takes.

**One fraction hides the failure mode.** A detector that fires on every noise
excursion recovers every true beat, so a recall-shaped score reports success
for an output that is unusable. Reporting sensitivity, PPV and F1 together
makes over-detection visible, and RMS timing jitter separates "found the beat"
from "found it in the right place" — which is the part that matters clinically,
since an interval measurement inherits the jitter of both endpoints.

The matching tolerance defaults to 150 ms, the ANSI/AAMI EC57 convention for
QRS-detector scoring against a reference annotation.

`detect_rpeaks` lazily imports `neurokit2`, following the `wfdb` pattern used
throughout `neuroflux_datasets`: nothing here pulls it in unless a caller asks
for detection, and the scoring half of the module has no dependency on it at
all. Passing `method` through to `neurokit2.ecg_peaks` means the detector can
be varied as a robustness check without touching the scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ANSI/AAMI EC57 matching window for QRS-detector scoring.
DEFAULT_TOLERANCE_MS = 150.0

# One beat interval at 60 bpm, excluded at each end of a scored window so a
# detector's warm-up is not counted as a missed beat. See `match_beats`.
DEFAULT_GUARD_MS = 1000.0


@dataclass(frozen=True)
class QrsScore:
    """Detector performance against a reference annotation.

    Attributes:
        n_reference: Reference beats in the scored window.
        n_detected: Beats the detector reported.
        true_positives: Reference beats matched one-to-one within tolerance.
        false_positives: Detections with no reference beat to match.
        false_negatives: Reference beats with no detection to match.
        sensitivity: `TP / n_reference`, the fraction of real beats recovered.
        ppv: `TP / n_detected`, the fraction of detections that were real.
        f1: Harmonic mean of `sensitivity` and `ppv`.
        rms_jitter_ms: Root-mean-square timing error over matched pairs, in
            milliseconds. `nan` when nothing matched.
        mean_offset_ms: Signed mean of `detected - reference` over matched
            pairs, in milliseconds — a systematic decoder delay shows up here
            and not in `rms_jitter_ms` alone. `nan` when nothing matched.
    """

    n_reference: int
    n_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    ppv: float
    f1: float
    rms_jitter_ms: float
    mean_offset_ms: float


def _ratio(numerator: int, denominator: int) -> float:
    """`numerator / denominator`, with an empty denominator scoring 0."""
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def match_beats(
    reference_samples: np.ndarray,
    detected_samples: np.ndarray,
    fs: float,
    tol_ms: float = DEFAULT_TOLERANCE_MS,
    guard_ms: float = 0.0,
    n_samples: int | None = None,
) -> QrsScore:
    """
    Score detected beats against a reference annotation, one-to-one.

    Every candidate pair within tolerance is considered, then pairs are
    accepted in order of increasing timing error, each reference beat and each
    detection being used at most once. Accepting by proximity rather than by
    position matters when two detections straddle one reference beat: a
    left-to-right greedy pass would take the first and leave the closer one to
    become a false positive.

    Args:
        reference_samples: Reference beat positions, in samples.
        detected_samples: Detected beat positions, in samples.
        fs: Sampling rate in Hz, used to convert the tolerance and the jitter.
        tol_ms: Matching window in milliseconds. The default follows
            ANSI/AAMI EC57.
        guard_ms: Exclude beats within this many milliseconds of either end of
            the window, on both sides of the comparison. A detector needs a few
            hundred milliseconds of context to declare the first beat, so on a
            short window its warm-up is otherwise scored as a miss: on MIT-BIH
            8 s windows that is one beat in ten, a 10-point sensitivity penalty
            that says nothing about the signal. Requires `n_samples`.
        n_samples: Length of the scored window in samples, needed to locate the
            trailing edge. Required when `guard_ms` is non-zero.

    Returns:
        A `QrsScore` for this window, over the beats that survive the guard.

    Raises:
        ValueError: If `fs` or `tol_ms` is not positive, if `guard_ms` is
            negative, if `guard_ms` is non-zero without `n_samples`, or if
            either input is not one-dimensional.
    """
    if fs <= 0.0:
        raise ValueError(f"fs must be positive, got {fs}")
    if tol_ms <= 0.0:
        raise ValueError(f"tol_ms must be positive, got {tol_ms}")
    if guard_ms < 0.0:
        raise ValueError(f"guard_ms must be non-negative, got {guard_ms}")
    if guard_ms > 0.0 and n_samples is None:
        raise ValueError("guard_ms needs n_samples to locate the window's end.")

    reference = np.sort(np.asarray(reference_samples, dtype=np.float64).ravel())
    detected = np.sort(np.asarray(detected_samples, dtype=np.float64).ravel())

    if np.asarray(reference_samples).ndim > 1 or np.asarray(detected_samples).ndim > 1:
        raise ValueError("Beat positions must be one-dimensional.")

    if guard_ms > 0.0 and n_samples is not None:
        guard = guard_ms * fs / 1000.0
        keep = (guard, float(n_samples) - guard)
        reference = reference[(reference >= keep[0]) & (reference <= keep[1])]
        detected = detected[(detected >= keep[0]) & (detected <= keep[1])]

    n_reference = int(reference.size)
    n_detected = int(detected.size)
    tol_samples = tol_ms * fs / 1000.0

    # Candidate pairs: for each reference beat, every detection inside the
    # tolerance window. searchsorted keeps this linear in the number of pairs
    # rather than quadratic in the number of beats.
    candidates: list[tuple[float, int, int]] = []
    if n_reference > 0 and n_detected > 0:
        lo = np.searchsorted(detected, reference - tol_samples, side="left")
        hi = np.searchsorted(detected, reference + tol_samples, side="right")
        for i in range(n_reference):
            for j in range(int(lo[i]), int(hi[i])):
                candidates.append((abs(detected[j] - reference[i]), i, j))

    candidates.sort()
    used_reference: set[int] = set()
    used_detected: set[int] = set()
    offsets: list[float] = []

    for _, i, j in candidates:
        if i in used_reference or j in used_detected:
            continue
        used_reference.add(i)
        used_detected.add(j)
        offsets.append(float(detected[j] - reference[i]))

    true_positives = len(offsets)
    sensitivity = _ratio(true_positives, n_reference)
    ppv = _ratio(true_positives, n_detected)
    denominator = sensitivity + ppv

    if offsets:
        offset_ms = np.asarray(offsets, dtype=np.float64) * 1000.0 / fs
        rms_jitter_ms = float(np.sqrt(float(np.mean(np.square(offset_ms)))))
        mean_offset_ms = float(np.mean(offset_ms))
    else:
        rms_jitter_ms = float("nan")
        mean_offset_ms = float("nan")

    return QrsScore(
        n_reference=n_reference,
        n_detected=n_detected,
        true_positives=true_positives,
        false_positives=n_detected - true_positives,
        false_negatives=n_reference - true_positives,
        sensitivity=sensitivity,
        ppv=ppv,
        f1=0.0 if denominator == 0.0 else 2.0 * sensitivity * ppv / denominator,
        rms_jitter_ms=rms_jitter_ms,
        mean_offset_ms=mean_offset_ms,
    )


def _neurokit() -> Any:
    """Import `neurokit2` on demand, with an actionable message if absent."""
    try:
        import neurokit2  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching
        raise ImportError(
            "detect_rpeaks needs neurokit2, which neuroflux-core does not "
            "require by default. Install it with `uv add neurokit2`, or pass "
            "reference and detected beat positions to match_beats directly."
        ) from exc
    return neurokit2


def detect_rpeaks(
    ecg: np.ndarray,
    fs: float,
    method: str = "neurokit",
    min_std: float = 1e-6,
) -> np.ndarray:
    """
    Detect R-peak positions in an ECG signal.

    Args:
        ecg: 1-D ECG signal.
        fs: Sampling rate in Hz.
        method: Detector passed through to `neurokit2.ecg_peaks` — `"neurokit"`
            by default, with `"pantompkins1985"` and `"hamilton2002"` among the
            alternatives, for varying the detector as a robustness check.
        min_std: Standard deviation at or below which the signal is treated as
            carrying no beats at all. A collapsed network output is a total
            failure, and reporting zero detections is the honest reading of it;
            without this guard the detector runs on numerical noise and invents
            beats from it.

    Returns:
        Sorted R-peak positions in samples, as an `int64` array. Empty when the
        signal is degenerate or the detector finds nothing.

    Raises:
        ValueError: If `ecg` is not one-dimensional or `fs` is not positive.
        ImportError: If `neurokit2` is not installed.
    """
    signal = np.asarray(ecg, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"ecg must be one-dimensional, got shape {signal.shape}.")
    if fs <= 0.0:
        raise ValueError(f"fs must be positive, got {fs}")

    if signal.size == 0 or not np.all(np.isfinite(signal)):
        return np.empty(0, dtype=np.int64)
    if float(np.std(signal)) <= min_std:
        return np.empty(0, dtype=np.int64)

    _, info = _neurokit().ecg_peaks(signal, sampling_rate=fs, method=method)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64).ravel()
    return np.sort(peaks)


def qrs_score(
    reference_samples: np.ndarray,
    ecg: np.ndarray,
    fs: float,
    tol_ms: float = DEFAULT_TOLERANCE_MS,
    method: str = "neurokit",
    guard_ms: float = DEFAULT_GUARD_MS,
) -> QrsScore:
    """
    Detect beats in `ecg` and score them against a reference annotation.

    Args:
        reference_samples: Reference beat positions, in samples, on the same
            time base as `ecg`.
        ecg: 1-D ECG signal to run the detector over — the reconstruction, or
            the unprocessed noisy signal when scoring the identity baseline.
        fs: Sampling rate in Hz.
        tol_ms: Matching window in milliseconds.
        method: Detector passed through to `neurokit2.ecg_peaks`.
        guard_ms: Edge exclusion in milliseconds; see `match_beats`. Defaults
            to a full beat interval at 60 bpm, which covers the detector's
            warm-up without discarding a meaningful part of a multi-second
            window. Pass 0.0 to score every beat.

    Returns:
        A `QrsScore` for this window.

    Raises:
        ValueError: If an argument fails the checks of `detect_rpeaks` or
            `match_beats`.
        ImportError: If `neurokit2` is not installed.
    """
    detected = detect_rpeaks(ecg, fs, method=method)
    return match_beats(
        reference_samples,
        detected,
        fs,
        tol_ms=tol_ms,
        guard_ms=guard_ms,
        n_samples=int(np.asarray(ecg).size),
    )
