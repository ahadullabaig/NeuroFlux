"""
One signal source for experiments, synthetic or real.

Mechanism experiments need to run the same pipeline on generated ECG and on
MIT-BIH records without branching everywhere. `load_clean_ecg` hides the
difference: both paths return a 1-D clean signal at the requested rate, so the
only thing a caller changes is a flag.

The real path goes through `mit_bih.annotator.extract_clean_segments`, which
already resamples to the target rate and normalises, and which excludes any
window carrying an artifact annotation. `wfdb` stays lazily imported behind it,
so nothing here pulls it in for the synthetic path.

**The record is the unit of analysis.** Per `docs/r2_findings.md`, statistics
over MIT-BIH pair on record, not on run — a helper that hands back one window
per record per seed is what makes that straightforward to write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroflux_datasets.synthetic import generate_ecg

# Where `mit_bih.downloader.download_record` writes by default.
DEFAULT_CACHE_DIR = "data/mit_bih"

# The 16 records used by the Phase 3 sweep, and the only ones whose SHA-256
# checksums are populated in the downloader manifest. See CLAUDE.md.
PHASE3_RECORDS: tuple[str, ...] = (
    "100",
    "101",
    "103",
    "105",
    "108",
    "112",
    "113",
    "119",
    "200",
    "207",
    "212",
    "214",
    "228",
    "231",
    "232",
    "234",
)


def load_reference_beats(
    unit: str,
    duration_s: float = 8.0,
    fs: float = 250.0,
    window: int = 0,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One clean MIT-BIH window together with its cardiologist beat annotations.

    There is no synthetic counterpart on purpose. `generate_ecg` builds its
    waveform from a known beat train, so "annotations" for it would be the
    generator's own parameters — scoring a detector against them measures the
    generator, not the pipeline. Ground truth here means a human marked it.

    Args:
        unit: MIT-BIH record id.
        duration_s: Window length in seconds.
        fs: Sampling rate of the returned signal, in Hz.
        window: Which artifact-free window to take, wrapped modulo the number
            available — the same indexing `load_clean_ecg` uses, so passing the
            same `window` to both returns the same window.
        cache_dir: Directory holding the cached `.dat`/`.hea`/`.atr` files.

    Returns:
        `(signal, beats)` — the 1-D clean window at `fs`, and the reference beat
        positions in samples relative to the window start.

    Raises:
        ValueError: If the record contains no artifact-free window of the
            requested duration.
        FileNotFoundError: If the record is not cached. Pre-cache with
            `mit_bih.downloader.download_record`.
    """
    _require_cached(unit, cache_dir)

    from neuroflux_datasets.mit_bih.annotator import (  # noqa: PLC0415
        extract_annotated_segments,
    )

    segments = extract_annotated_segments(unit, cache_dir, duration_s, fs=fs)
    if not segments:
        raise ValueError(
            f"Record {unit} has no artifact-free {duration_s}s window. "
            f"Shorten the window or drop the record."
        )
    chosen = segments[window % len(segments)]
    return chosen.signal, chosen.beats


def _require_cached(unit: str, cache_dir: str | Path) -> None:
    """Raise a message naming the fix if a record has not been downloaded."""
    header = Path(cache_dir) / f"{unit}.hea"
    if not header.exists():
        raise FileNotFoundError(
            f"Record {unit} is not cached at {cache_dir}. Pre-cache it with "
            f"download_record({unit!r}, {str(cache_dir)!r})."
        )


def load_clean_ecg(
    dataset: str,
    unit: str = "synthetic",
    duration_s: float = 8.0,
    fs: float = 250.0,
    window: int = 0,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """
    One clean ECG signal for one analysis unit.

    Args:
        dataset: `"synthetic"` for generated ECG, `"mit-bih"` for a real record.
        unit: Record id when `dataset` is `"mit-bih"`; ignored otherwise.
        duration_s: Window length in seconds.
        fs: Sampling rate of the returned signal, in Hz.
        window: Which artifact-free window to take, wrapped modulo the number
            available. A seed index can be passed straight in — it selects a
            different real window per seed rather than reusing one.
        cache_dir: Directory holding the cached `.dat`/`.hea`/`.atr` files.

    Returns:
        A 1-D float array of clean signal at `fs`.

    Raises:
        ValueError: If `dataset` is unknown, or if the record contains no
            artifact-free window of the requested duration.
        FileNotFoundError: If the record is not cached. Pre-cache with
            `mit_bih.downloader.download_record`.
    """
    if dataset == "synthetic":
        return generate_ecg(duration_s=duration_s, fs=fs)

    if dataset != "mit-bih":
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'synthetic' or 'mit-bih'."
        )

    _require_cached(unit, cache_dir)

    from neuroflux_datasets.mit_bih.annotator import (  # noqa: PLC0415
        extract_clean_segments,
    )

    segments = extract_clean_segments(unit, cache_dir, duration_s, fs=fs)
    if not segments:
        raise ValueError(
            f"Record {unit} has no artifact-free {duration_s}s window. "
            f"Shorten the window or drop the record."
        )
    return segments[window % len(segments)]
