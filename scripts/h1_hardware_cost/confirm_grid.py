"""
H1 confirmation grid — the paper's headline designs on data no H1 test has seen.

The MCSoC paper's headline comparisons came partly post hoc (`docs/h1_findings.md`
§3.2: `[512,128]` K = 8 against R4b's `[512,512]` K = 16 used B2's 5 seeds
against R4b's 10). This grid re-measures every design the paper names, on
seeds 10-19. For MIT-BIH, `load_signal` takes a different artifact-free window
per seed and every record has at least 103 of them, so seeds 10-19 are ECG
windows no earlier grid used. It adds realistic input SNRs and two further
signals.

Designs, all rate-controlled and untrained, fixed fan-in in the first layer:

  D0  shipped architecture   [32,16,8] dense, layers 40;40 Hz, encoder 40 Hz
  D1  previous best          [512,512] K = 16, layer 20 Hz, encoder 40 Hz
  D2  headline               [512,128] K = 8,  layer 20 Hz, encoder 40 Hz
  D3  headline, 2-bit        D2 with weights quantized to 2 bits
  D4  wide pool, slow encoder [2048,128] K = 8, layer 20 Hz, encoder 10 Hz
  D5  narrow pool            [128,128] K = 8,  layer 20 Hz, encoder 40 Hz

Signals: MIT-BIH (16 records), synthetic EEG and the 2 Hz sine (unit = seed).
Inputs: noiseless, 10 dB and 0 dB Gaussian interference. Seeds 10-19.
6 designs x (16 + 2) units x 10 seeds x 3 inputs = 3,240 runs.

Written before the grid ran; no row of it has been seen. The unit is a record
on MIT-BIH and a seed on the synthetic signals; SNR averages a unit's `ok` runs;
paired tests pair units; equivalence is TOST at +/-0.5 dB; intervals are BCa.
A claim that needs several tests to all pass is an intersection-union test and
takes no multiplicity adjustment.

Primary, MIT-BIH, noiseless input (n = 16):

  C1  Headline. D2 minus D1 is equivalent.
  C2  Two bits. D3 minus D2 is equivalent.
  C3  Break-even pair. D4 minus D5 is non-inferior: the lower BCa bound of the
      paired median difference is above -0.5 dB.
  C4  Costs, from measured rates, not a statistical test. D2 uses at most
      0.13x D1's synaptic operations and fewer spikes. D4 uses at most 0.26x
      D5's synaptic operations and more spikes, so the pair has a finite
      break-even ratio of spike to synaptic-operation energy, which is
      reported: (synops_D5 - synops_D4) / (spikes_D4 - spikes_D5).

Secondary, MIT-BIH, noisy input:

  C5  D2 minus D1 is equivalent at 10 dB and at 0 dB input (both must hold).

Generality, synthetic EEG and sine, noiseless input (n = 10 seeds each):

  G1  D2 minus D1 is non-inferior on each signal (both must hold).
  G2  Descriptive: every design's ceiling and costs per signal and input.

Reading rule for the paper.
- C1 and C2 hold: the headline and abstract say the design matches the best
  earlier ceiling with 2-bit weights, quoting this grid's numbers.
- C1 fails: the paper states the measured difference at 1/8 the synaptic
  work, and does not say "matches".
- C3 holds: the break-even example uses this pair and its measured ratio.
- C5 fails at an input: the headline is stated for the inputs where it holds.
- G1 fails on a signal: the headline is scoped to ECG, and says so.

Usage:
    OMP_NUM_THREADS=1 uv run python scripts/h1_hardware_cost/confirm_grid.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from r3_common import CLEAN, Spec, run_grid  # noqa: E402

SEEDS: tuple[int, ...] = tuple(range(10, 20))
INPUT_SNRS_DB: tuple[float, ...] = (CLEAN, 10.0, 0.0)
SYNTHETIC_SIGNALS: tuple[str, ...] = ("eeg", "sine")
OUTPUT = Path("results/h1_hardware_cost/confirm_grid.csv")

# name -> (layers, layer rates, encoder Hz, fan-in, weight bits)
DESIGNS: dict[
    str, tuple[tuple[int, ...], tuple[float, ...], float, int | None, int | None]
] = {
    "D0": ((32, 16, 8), (40.0, 40.0), 40.0, None, None),
    "D1": ((512, 512), (20.0,), 40.0, 16, None),
    "D2": ((512, 128), (20.0,), 40.0, 8, None),
    "D3": ((512, 128), (20.0,), 40.0, 8, 2),
    "D4": ((2048, 128), (20.0,), 10.0, 8, None),
    "D5": ((128, 128), (20.0,), 40.0, 8, None),
}


def units() -> list[tuple[str, str]]:
    return [("mit-bih", record) for record in PHASE3_RECORDS] + [
        (signal, "synthetic") for signal in SYNTHETIC_SIGNALS
    ]


def spec(design: str, signal: str, unit: str, seed: int, snr_db: float) -> Spec:
    layers, rates, encoder_hz, fan_in, bits = DESIGNS[design]
    return Spec(
        experiment="h1-confirm",
        signal=signal,
        unit=unit,
        seed=seed,
        input_snr_db=snr_db,
        layers=layers,
        layer_rates_hz=rates,
        encoder_rate_hz=encoder_hz,
        fan_in=fan_in,
        weight_bits=bits,
    )


def specs() -> list[Spec]:
    return [
        spec(design, signal, unit, seed, snr_db)
        for signal, unit in units()
        for seed in SEEDS
        for snr_db in INPUT_SNRS_DB
        for design in DESIGNS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    grid = specs()
    print(f"H1 confirmation grid: {len(grid)} runs -> {OUTPUT}")
    run_grid(grid, OUTPUT, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
