"""
R5a — is the ceiling specific to rate coding, or to the population decoder?

Finding 5 in `docs/paper_roadmap.md` compared encoders and decoders on trained,
uncalibrated networks, left the delta rows degenerate, and read a
time-to-first-spike code with rate decoders that cannot see it. This grid
re-measures the encoder x decoder table at R3's operating point.

**Grid.** Rate-controlled, untrained `[32,16,8]`; hidden layer at 40 Hz, output
at 40 Hz or 5 Hz (the MIT-BIH rate optimum, R3 C3). Encoders, all at 40 Hz:

  poisson   calibrated ceiling, as every R3 grid;
  temporal  time-to-first-spike, 24 ms window (41.7 Hz, six latency levels);
  delta     absolute mode, threshold calibrated to 40 Hz on each drive
            (`calibrate_delta_threshold`; roadmap next action 12).

MIT-BIH (16 records), synthetic ECG, EEG and the 2 Hz sine; 10 seeds; Gaussian
noise at -10, 0, 10, 20, 30 and 40 dB, and noiseless. Every stage is read with
population decoders (tau 10, 20, 40 ms), spike-rate decoders (trailing window
20, 40, 80 ms) and, for the temporal encoder, a latency readout
(`r3_common.latency_readout`). A boxcar of W ms has the noise bandwidth of an
exponential of W/2 ms, so the three spike-rate windows pair with the three
population time constants.

**What a pilot showed before this grid was fixed** (6 units, seed 0, not
analysed further): rate readouts of the temporal code score about 0 dB; its
latency readout reaches 5.7 dB on the sine at a 40 Hz output but trails the
signal by about 40 ms, which on its own caps a 2 Hz sine near 6.5 dB. Readouts
differ in delay by that much, so comparing encoders on causal SNR would compare
delays.

**Metrics.** Primary for every comparison below: `snr_best_lag_db`, affine SNR
after advancing the output by its best lag in 0-64 ms (positive scale only).
Reported alongside: `snr_affine_db`, the causal SNR of every R3 table. Unit: a
record on MIT-BIH, a seed otherwise; SNR is averaged over `ok` runs.

**Best readout.** For each (signal, output rate, encoder), the readout with the
highest median-over-units noiseless ceiling on the primary metric. It is chosen
once and used at every input.

**Analysis, fixed before the grid was run.**

  T1  Does the ceiling disappear under temporal coding? At a 40 Hz output, per
      signal, temporal minus Poisson noiseless ceiling (best readouts), paired
      by unit: Wilcoxon, Holm across the four signals, BCa interval on the
      median difference, TOST +-0.5 dB. Temporal coding *escapes* the ceiling
      on a signal if the difference is significant with a median of at least
      +3 dB (twice the S/N). The ceiling is specific to rate coding only if
      temporal coding escapes on at least three of the four signals. The same
      comparison at 5 Hz, and delta minus Poisson, are reported but do not
      enter the decision.
  T2  Flatness per encoder: per unit, the OLS slope of output SNR (best readout)
      on input SNR over 10-40 dB; median over units with a BCa interval. Flat
      if the interval's upper bound is below 1, C1's criterion. Both rates.
  T3  Decoder type at matched bandwidth, Poisson only: spike-rate W minus
      population W/2 for W = 20, 40, 80, and best spike-rate minus best
      population, paired by unit on the noiseless ceiling, TOST +-0.5 dB, per
      signal and rate. The ceiling is not a population-decoder artefact if no
      spike-rate readout is significantly *higher* than its matched population
      readout by more than 0.5 dB.
  T4  Delta at a calibrated threshold: at least 95% of runs `ok`, which closes
      next action 12. Its ceilings are described, not tested.

  Check: the Poisson population tau = 20 causal SNR must equal R3 C1's
  controlled arm at every shared (unit, seed, input); `analyse_r5` refuses to
  report otherwise.

Usage:
    uv run python scripts/r5_generality/encoders_decoders.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from r3_common import CLEAN, DENSE_RATE_HZ, Spec, run_grid  # noqa: E402

INPUTS: tuple[float, ...] = (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, CLEAN)
OUTPUT_RATES: tuple[float, ...] = (DENSE_RATE_HZ, 5.0)
RATE_READOUTS: tuple[tuple[str, float], ...] = (
    ("population", 10.0),
    ("population", 20.0),
    ("population", 40.0),
    ("spike-rate", 20.0),
    ("spike-rate", 40.0),
    ("spike-rate", 80.0),
)
LATENCY: tuple[str, float] = ("latency", 0.0)
DEFAULT_OUT = Path("results/r5_generality/encoders_decoders.csv")


def readouts_for(encoder: str | None) -> tuple[tuple[str, float], ...]:
    """Every readout a run with this encoder is scored with."""
    return RATE_READOUTS + ((LATENCY,) if encoder == "temporal" else ())


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    units = [
        ("synthetic-ecg", "synthetic"),
        ("eeg", "synthetic"),
        ("sine", "synthetic"),
    ] + [("mit-bih", r) for r in records]
    return [
        Spec(
            experiment="r5-encoders",
            signal=signal,
            unit=unit,
            seed=seed,
            input_snr_db=snr,
            layer_rates_hz=(DENSE_RATE_HZ, rate),
            encoder=encoder,
            readouts=readouts_for(encoder),
        )
        for signal, unit in units
        for seed in range(seeds)
        for encoder in (None, "temporal", "delta")
        for rate in OUTPUT_RATES
        for snr in INPUTS
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
    print(f"R5a encoders x decoders: {len(specs)} runs -> {args.out}")
    run_grid(specs, args.out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
