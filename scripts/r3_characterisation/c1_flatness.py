"""
R3 C1 — is the ceiling flat in input quality?

Finding 1 measured output SNR against input SNR on synthetic ECG with three
seeds, and its input axis was not what it claimed: flicker at a fixed 0.5
amplitude was added before the noise, so no nominal setting could lift the
input past about −6 dB against the clean signal. This grid re-measures the
curve on an axis that spans what it says, at the roadmap's power.

**Grid.**

  signals        synthetic ECG, the 16 Phase 3 MIT-BIH records, EEG, sine
  input SNR      −20 to +40 dB in 5 dB steps, plus a noiseless input
  seeds          10 (MIT-BIH: a different artifact-free window per seed)
  arms           controlled passive  — the operating point M3 derived the
                                       ceiling on (scaled init, encoder and
                                       both layers held at 40 Hz, no training)
                 shipped passive     — `cli.py`'s network before learning
                 shipped trained     — `cli.py`'s network after 10 STDP epochs

**Interference type.** Flicker is not dropped; it is compared at matched
input SNR. On synthetic ECG and MIT-BIH, at −10, −5, 0, 5, 10 and 20 dB, the
controlled-passive and shipped-trained arms also run with a 50 Hz flicker and
with an equal-power flicker/noise mix, each scaled by
`synthetic.add_interference` to the exact target SNR.

**Analysis, fixed before the grid was run.**

  * The unit is a record on MIT-BIH and a seed on the synthetic signals. A
    record's value at each input SNR is the mean over its seeds whose verdict
    is `ok`; the fraction that were not is reported beside it.
  * Flatness is the least-squares slope of affine-corrected output SNR on input
    SNR, fitted per unit over two ranges: the full −20 to +40 dB, and the high
    range 10 to 40 dB where the unprocessed input's own slope is 1. The slope's
    median across units gets a BCa bootstrap interval. **Acceptance: the
    interval excludes 1.0 on all four signals** (roadmap R3).
  * The unprocessed input's affine SNR over the same range is the reference a
    lossless pipeline would track.
  * The cascade law is tested with no fitted parameters: each unit's ceiling is
    its own noiseless-input output, and the output at every other input is
    predicted as `cascade_snr_db(input, [ceiling])`. Residuals are reported
    separately at input SNR >= 10 dB and below, as in M3 §8.
  * Interference type: flicker minus Gaussian and mix minus Gaussian, paired by
    unit at each input SNR; Wilcoxon, Holm across the input SNRs within each
    (signal, arm, type) family, TOST at the pre-registered ±0.5 dB.

Usage:
    uv run python scripts/r3_characterisation/c1_flatness.py
    uv run python scripts/r3_characterisation/c1_flatness.py --seeds 2 --records 100 214
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CLEAN, Spec, run_grid  # noqa: E402

INPUT_SNRS: tuple[float, ...] = tuple(float(v) for v in range(-20, 45, 5))
INTERFERENCE_SNRS: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
# (operating point, epochs)
ARMS: tuple[tuple[str, int], ...] = (
    ("controlled", 0),
    ("shipped", 0),
    ("shipped", 10),
)
INTERFERENCE_ARMS: tuple[tuple[str, int], ...] = (("controlled", 0), ("shipped", 10))
DEFAULT_OUT = Path("results/r3_characterisation/c1_flatness.csv")


def units_for(signal: str, records: list[str]) -> list[str]:
    """The analysis units of one signal type."""
    return list(records) if signal == "mit-bih" else ["synthetic"]


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    """Every run in the C1 grid, with the interference-type control included."""
    specs: list[Spec] = []
    for signal in ("synthetic-ecg", "mit-bih", "eeg", "sine"):
        for unit in units_for(signal, records):
            for seed in range(seeds):
                for operating_point, epochs in ARMS:
                    for snr in (*INPUT_SNRS, CLEAN):
                        specs.append(
                            Spec(
                                experiment="c1",
                                signal=signal,
                                unit=unit,
                                seed=seed,
                                input_snr_db=snr,
                                operating_point=operating_point,
                                epochs=epochs,
                            )
                        )
                if signal not in ("synthetic-ecg", "mit-bih"):
                    continue
                for operating_point, epochs in INTERFERENCE_ARMS:
                    for kind in ("flicker", "mixed"):
                        for snr in INTERFERENCE_SNRS:
                            specs.append(
                                Spec(
                                    experiment="c1",
                                    signal=signal,
                                    unit=unit,
                                    seed=seed,
                                    input_snr_db=snr,
                                    interference=kind,
                                    operating_point=operating_point,
                                    epochs=epochs,
                                )
                            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = build_specs(args.seeds, list(args.records))
    print(f"R3 C1 — flatness: {len(specs)} runs -> {args.out}")
    run_grid(specs, args.out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
