"""
R4b (roadmap next action 14) — can the encoder's range recover what noise costs it?

R3 C1 §1.5 showed that the operating point a noisy input sets costs the ceiling
1.11 dB at 0 dB input on MIT-BIH and 5.84 dB on the sine, because the encoder's
`[v_min, v_max]` is the noisy drive's own minimum and maximum: noise widens the
range and squeezes the signal into less of it. This grid replaces that choice.

  minmax      the drive's extremes — R3's encoder, the reference
  percentile  its 0.1st / 99.9th percentiles
  lowpass     the extremes of a zero-phase 40 Hz low-pass copy of the drive
  oracle      the clean signal's extremes — not deployable; the bound

The encoder still sees the unfiltered noisy drive in every arm; only its range
changes, so samples outside the range clip. That clip is a limiter, which is
exactly the nonlinearity M3 wrongly assumed was already present.

**Grid.** Rate-controlled, untrained `[32,16,8]` (encoder and both layers at
40 Hz); MIT-BIH (16 records), synthetic ECG and the 2 Hz sine; 10 seeds; input
SNR −10, −5, 0, 5, 10, 20 dB and noiseless; Gaussian noise.

**Analysis, fixed before the grid was run.**

  * Unit: a record on MIT-BIH, a seed otherwise; SNR over `ok` runs.
  * Each method is paired with `minmax` by unit at every input SNR: Wilcoxon,
    Holm across the seven inputs within (signal, method), BCa interval on the
    median difference, TOST ±0.5 dB.
  * **Primary test:** on MIT-BIH at 0 dB input, a *deployable* method
    (`percentile` or `lowpass`) raises output SNR significantly, by a median of
    at least +0.5 dB, **and** its noiseless ceiling is equivalent to or higher
    than `minmax`'s. Both conditions, or the lever is not established.
  * The share of the oracle's gain each deployable method recovers is reported
    as a description, at every input.

Usage:
    uv run python scripts/r4b_mechanism/encoder_range.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from r3_common import CLEAN, ENCODER_RANGES, Spec, run_grid  # noqa: E402

INPUTS: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 20.0, CLEAN)
DEFAULT_OUT = Path("results/r4b_mechanism/encoder_range.csv")


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    units = [("synthetic-ecg", "synthetic"), ("sine", "synthetic")] + [
        ("mit-bih", r) for r in records
    ]
    return [
        Spec(
            experiment="r4b-encoder-range",
            signal=signal,
            unit=unit,
            seed=seed,
            input_snr_db=snr,
            encoder_range=method,
        )
        for signal, unit in units
        for seed in range(seeds)
        for snr in INPUTS
        for method in (None, *ENCODER_RANGES)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = build_specs(args.seeds, list(args.records))
    print(f"R4b encoder range: {len(specs)} runs -> {args.out}")
    run_grid(specs, args.out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
