"""
R3 §1.6 — two checks on results that predate R3, made while verifying C1's confound.

Descriptive measurements, not pre-registered tests.

  1. The true input SNR of the Phase 3 / R1 / R2 MIT-BIH pairs. `make_pair` adds
     flicker at 0.5 amplitude before scaling Gaussian noise to the flickered
     signal's power, so the labelled SNR is not the SNR against the clean signal.
  2. The QRS "doing nothing" baseline with and without neurokit's standard
     `ecg_clean`. `metrics/qrs.detect_rpeaks` passes the raw signal to
     `neurokit2.ecg_peaks`, which never removes the flicker.

Usage:
    uv run python scripts/r3_characterisation/upstream_checks.py
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_datasets.mit_bih.annotator import make_pair
from neuroflux_datasets.signals import (
    DEFAULT_CACHE_DIR,
    PHASE3_RECORDS,
    load_reference_beats,
)

FS = 250.0
NOMINAL_SNRS: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    import neurokit2 as nk  # type: ignore[import-untyped]  # noqa: PLC0415

    print(
        f"{len(PHASE3_RECORDS)} records x {args.seeds} seeds, {args.duration:g} s, "
        "50 Hz flicker at 0.5 amplitude (the Phase 3 / R2 configuration)\n"
    )
    print(
        f"{'nominal':>8} | {'true SNR median':>15} {'range':>17} | "
        f"{'F1 raw':>7} {'F1 ecg_clean':>12}"
    )
    for nominal in NOMINAL_SNRS:
        true_snrs: list[float] = []
        raw: list[float] = []
        cleaned: list[float] = []
        for record in PHASE3_RECORDS:
            record_raw: list[float] = []
            record_clean: list[float] = []
            for seed in range(args.seeds):
                clean, noisy = make_pair(
                    record, args.cache_dir, args.duration, 50.0, 0.5, nominal, seed=seed
                )
                signal, beats = load_reference_beats(
                    record, args.duration, FS, window=seed, cache_dir=args.cache_dir
                )
                if not np.allclose(signal, clean):
                    raise RuntimeError(
                        f"window mismatch on record {record} seed {seed}"
                    )
                true_snrs.append(calculate_snr(clean, noisy))
                record_raw.append(qrs_score(beats, noisy, FS).f1)
                filtered = nk.ecg_clean(noisy, sampling_rate=int(FS), method="neurokit")
                record_clean.append(qrs_score(beats, np.asarray(filtered), FS).f1)
            raw.append(float(np.mean(record_raw)))
            cleaned.append(float(np.mean(record_clean)))
        print(
            f"{nominal:>+8.1f} | {statistics.median(true_snrs):>+15.2f} "
            f"[{min(true_snrs):+.2f}, {max(true_snrs):+.2f}] | "
            f"{statistics.median(raw):>7.3f} {statistics.median(cleaned):>12.3f}"
        )
    print("\nF1 columns: median over records of each record's mean over seeds.")


if __name__ == "__main__":
    main()
