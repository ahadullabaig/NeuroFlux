"""
H1 grid B2 — the two cost levers pilot B could not see.

Pilot B (`docs/h1_findings.md` §1) put a hardware cost on every committed
design point and found the grids never varied the two quantities that should
set cost: the encoder rate, which drives 80-100% of the synaptic operations,
and the width of the input pool behind a fixed fan-in. This grid varies both.

Design: one LIF layer `[N_in, N_1]` held at 20 Hz (R4b's rate optimum for
one-layer networks), every neuron drawing exactly K = 8 of the N_in tiled input
trains; N_1 in {32, 128}; N_in in {32, 128, 512, 2048}; encoder held at 10, 20
or 40 Hz. Rate-controlled, untrained, noiseless input, 16 MIT-BIH records x 5
seeds: 24 design points, 1,920 runs.

At a fixed fan-in the synaptic operations are r_0 * K * N_1, independent of
N_in, so the input pool is a lever that costs encoder spikes (N_in * r_0
address events) and nothing else. Halving the encoder rate halves the synaptic
work.

Written before the grid ran. The unit is a record; SNR averages a record's `ok`
runs; paired tests pair records (Wilcoxon, Holm where a family is named, BCa
intervals), as in R3.

  P1  Input pool. At the 40 Hz encoder, the per-record slope of ceiling on
      log2(N_in) is positive at N_1 = 32 and at N_1 = 128 (Wilcoxon against
      zero, Holm across the two). Rationale: at fixed K the column cosine
      3K/4N_in falls as the pool widens, so neurons share less input noise
      while each still averages K trains (M4, R4b).

  P2a Encoder rate. At each of the 8 (N_in, N_1) architectures, the ceiling
      with the encoder at 40 Hz exceeds the ceiling at 10 Hz (Holm across the
      8). Reported as dB lost per halving of the encoder rate.

  P2b The cascade law prices the encoder rate. A Poisson encoder's counting
      noise gives it an inverse S/N proportional to 1/r_0, so in S/N terms the
      law reads 1/R_out = c / r_0 + 1/Lambda_rest, with c and Lambda_rest fixed
      per record and architecture. Both are solved from the 20 Hz and 40 Hz
      cells, and the law then predicts the ceiling at 10 Hz. Supported if the
      median absolute error over all 8 x 16 cells is at most 0.3 dB (affine).
      A cell whose predicted 1/R_out is not positive counts as an unbounded
      error. If P2b fails, the layer's own noise depends on its input rate,
      an operating-point effect of the kind R3 §1.5 found for input SNR.

      Amended while the grid was running, before any of its rows were
      analysed. As first written, P2b backed out the layer's intrinsic S/N
      from the encoder stage's read-out SNR. Three smoke-test runs showed the
      layer's output SNR above that read-out's (3.29 against 2.83 dB at
      `[128,128]`, 40 Hz), where `intrinsic_snr_db` returns inf and the test
      is empty: the stage read-out does not measure what reaches the layer,
      whose membrane averages noise the read-out leaves in. The 1/r_0 form
      needs no stage read-out.

Frame decision (pilot B's rule, `docs/h1_findings.md` §2). Cost against
fidelity becomes the MCSoC paper's frame if either holds:

  F1  At N_1 = 128 and the 40 Hz encoder, N_in = 2048 minus N_in = 128 is at
      least +0.5 dB, paired, with Wilcoxon p < 0.05.
  F2  A B2 point dominates `[128,128]`, K = 8, 40 Hz encoder (this grid's own
      rows). Candidates are fixed by nominal cost before any ceiling is read:
      non-inferiority for the 20 points at no more than half its synaptic
      operations (BCa lower bound of the paired difference above -0.5 dB),
      superiority for the 3 other points at the same synaptic operations
      (paired difference > 0, Holm p < 0.05 across the 3).

Check: the `[128,128]`, K = 8, 40 Hz encoder cell repeats R4b's fan-in cell at
20 Hz for seeds 0-4 with identical random streams, so its ceilings must match
`results/r4b_mechanism/fan_in.csv` exactly.

Usage:
    uv run python scripts/h1_hardware_cost/b2_grid.py
    uv run python scripts/h1_hardware_cost/analyse_b2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from r3_common import CLEAN, Spec, run_grid  # noqa: E402

FAN_IN = 8
LAYER_RATE_HZ = 20.0
LAYER_WIDTHS: tuple[int, ...] = (32, 128)
INPUT_WIDTHS: tuple[int, ...] = (32, 128, 512, 2048)
ENCODER_RATES_HZ: tuple[float, ...] = (10.0, 20.0, 40.0)
OUTPUT = Path("results/h1_hardware_cost/b2_grid.csv")


def specs(seeds: int, records: list[str]) -> list[Spec]:
    return [
        Spec(
            experiment="h1-b2",
            signal="mit-bih",
            unit=record,
            seed=seed,
            input_snr_db=CLEAN,
            layers=(n_in, n_1),
            layer_rates_hz=(LAYER_RATE_HZ,),
            encoder_rate_hz=encoder_hz,
            fan_in=FAN_IN,
        )
        for record in records
        for seed in range(seeds)
        for n_1 in LAYER_WIDTHS
        for n_in in INPUT_WIDTHS
        for encoder_hz in ENCODER_RATES_HZ
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    grid = specs(args.seeds, list(args.records))
    print(f"H1 grid B2: {len(grid)} runs -> {OUTPUT}")
    run_grid(grid, OUTPUT, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
