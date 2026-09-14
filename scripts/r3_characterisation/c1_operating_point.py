"""
R3 C1, post hoc — does noise move the operating point the cascade law assumes fixed?

**This diagnostic was designed after C1's results were seen.** It is not part
of C1's pre-registered analysis and is reported separately from it.

What C1 found. Applied in `S/N` terms, the cascade law removes the ~2 dB
over-prediction of loss that applying it to affine SNR produced, but it leaves
a structured residual: measured output falls 0.3–0.6 dB *below* prediction
between 0 and 10 dB input on ECG, and up to ~2 dB below it on the sine. The
law takes each stage's intrinsic SNR from one noiseless run and assumes it
holds at every input. That assumption fails if the noise changes the operating
point, and in this pipeline it can: the encoder's range is the noisy drive's
own minimum and maximum, so noise widens it and compresses the signal into a
smaller share of the encoder's rate range; the encoder-rate and layer-gain
calibrations are then fitted to that noisy drive.

The test. For every input SNR `s`, run the network on the **noiseless** signal
but take its whole operating point — encoder range, encoder rate calibration,
layer gains — from the same seed's drive at `s` dB. That run's output is the
ceiling *at the operating point an s dB input sets*. Predict the noisy run at
`s` from it instead of from the ordinary noiseless run.

  * If the residual collapses, the law is exact once its intrinsic SNRs are
    measured at the right operating point, and the 0–10 dB shortfall is range
    compression, not a failure of additivity.
  * If it does not, something else — a nonlinearity in how noise and signal
    interact inside the LIF layers — is responsible, and the law is only an
    upper bound below 10 dB.

Grid: C1's signals, units, seeds and input SNRs; the untrained arms only
(rate-controlled and shipped), since a trained network's weights also depend on
the input and would confound the operating point with learning. Noiseless
input throughout. The noisy runs it is compared against are C1's own rows.
The rate-controlled arm on both ECG signals is also calibrated at C1's flicker
inputs, which asks whether flicker's smaller cost (C1 §1.4) is the same effect.

Usage:
    uv run python scripts/r3_characterisation/c1_operating_point.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_flatness import INPUT_SNRS, INTERFERENCE_SNRS, units_for  # noqa: E402
from r3_common import CLEAN, Spec, run_grid  # noqa: E402

ARMS: tuple[tuple[str, int], ...] = (("controlled", 0), ("shipped", 0))
DEFAULT_OUT = Path("results/r3_characterisation/c1_operating_point.csv")


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    """Noiseless runs, each calibrated at one of C1's noisy inputs."""
    specs = [
        Spec(
            experiment="c1-operating-point",
            signal=signal,
            unit=unit,
            seed=seed,
            input_snr_db=CLEAN,
            operating_point=operating_point,
            epochs=epochs,
            calibrate_at_snr_db=snr,
        )
        for signal in ("synthetic-ecg", "mit-bih", "eeg", "sine")
        for unit in units_for(signal, records)
        for seed in range(seeds)
        for operating_point, epochs in ARMS
        for snr in INPUT_SNRS
    ]
    specs += [
        Spec(
            experiment="c1-operating-point",
            signal=signal,
            unit=unit,
            seed=seed,
            input_snr_db=CLEAN,
            interference="flicker",
            calibrate_at_snr_db=snr,
        )
        for signal in ("synthetic-ecg", "mit-bih")
        for unit in units_for(signal, records)
        for seed in range(seeds)
        for snr in INTERFERENCE_SNRS
    ]
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
    print(f"R3 C1 operating-point diagnostic: {len(specs)} runs -> {args.out}")
    run_grid(specs, args.out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
