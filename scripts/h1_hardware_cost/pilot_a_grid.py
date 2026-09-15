"""
H1 pilot A — what weight precision and device faults cost, and whether the
B2 noise budget prices them.

B2 (`docs/h1_findings.md` §3) split a fixed-fan-in layer's output noise into
an encoder counting term and a rate-independent remainder,
`1/R_out = c/r_0 + 1/Lambda_rest` in `S/N` terms, fitted at encoder rates of
20 and 40 Hz and confirmed at 10 Hz to 0.07 dB. A neuron's drive averages its
input trains' counting noise over its effective fan-in, the participation
ratio `PR = (sum w)^2 / sum w^2` (R4b §1), so anything that lowers PR should
raise the counting term by `PR_exact / PR_realised` and leave the remainder
alone. Weight quantization, device-to-device variation and open (stuck-at-zero)
devices all lower PR by amounts computable from the weight matrices before any
simulation. That is a zero-parameter prediction, which this grid tests.

Design. The three B2 networks with `N_1` = 128: `[128,128]`, `[512,128]` and
`[2048,128]`, K = 8, the layer held at 20 Hz; encoder at 10, 20 and 40 Hz;
noiseless input; 16 MIT-BIH records x 5 seeds. Twelve device levels, one
non-ideality at a time (`r3_common.perturb_weights`):

  exact                          B2's own networks
  quantized                      1, 2, 3, 4 bits on [0, max w]
  lognormal variation            sigma 0.25, 0.5, 1.0, 1.5
  stuck-at-zero                  25%, 50%, 75% of devices

3 networks x 3 rates x 12 levels = 108 design points, 8,640 runs. The
predicted change at every point, from B2's per-record fits and the matrices'
PR, is written by `pilot_a_predictions.py` to
`results/h1_hardware_cost/analysis/pilot_a_predictions.csv` and committed
before this grid runs.

Written before the grid ran. The unit is a record; SNR averages a record's `ok`
runs; paired tests pair records, as in R3; equivalence is TOST at +/-0.5 dB.

  PA1 Precision budget. For each network and encoder rate, b_min is the fewest
      bits whose ceiling, and every larger bit width's, is equivalent to exact
      weights. Prediction: b_min <= 2 at all 9 (the budget predicts at most
      0.07 dB lost at 2 bits).

  PA2 Variation budget. sigma_max is the largest sigma equivalent to exact
      weights with every smaller sigma equivalent too. Prediction:
      sigma_max >= 0.5 at all 9 (at most 0.17 dB predicted at 0.5).

  PA3 The budget prices the faults. Per record and design point,
      `1/R_pred = c * (PR_exact / PR_level) / r_0 + 1/Lambda_rest`, with c and
      Lambda_rest from that record's B2 cells at 20 and 40 Hz and PR averaged
      over the 5 seeds' matrices. Supported if both hold: (i) the median
      absolute error over all 99 perturbed design points x 16 records is at
      most 0.3 dB; (ii) over the design points where the median predicted
      loss is at least 0.5 dB, the model's median absolute error is below
      that of the null model, which predicts no change from exact weights.
      (ii) exists because most levels are predicted to cost little, where the
      null is also right.

  PA4 Is effective fan-in the whole story? Two level pairs have matched PR but
      very different spread in each neuron's total weight: 50% stuck against
      sigma 1.5 (PR 3.14-3.16 against 3.15-3.21; coefficient of variation of the
      summed weight 0.44 against 1.02-1.23), and 1 bit against sigma 1.0 (PR
      3.99 against 4.01-4.06; 0.36 against 0.54-0.60). The layer gain is one
      number, so a wide spread leaves some neurons driven far harder than
      others. Prediction if PR is the whole story: each pair equivalent at all
      9 network x rate points (18 TOST tests). A pair that is not says
      heterogeneity costs something PR does not see.

Design points where fewer than half the runs are `ok` are excluded from PA3 and
counted. At 75% stuck about 9% of neurons lose every input; PR is averaged over
neurons with any input, which is a known gap in the model.

Reading rule for the paper. PA3 supported: the MCSoC paper's budget prices
encoder rate, input pool, weight precision and device faults with one model.
PA3 not supported: PA1 and PA2 give an empirical precision and fault table and
the model claim stays with the encoder. Either way the design adds weight
memory (bits x synapses) as a cost.

Disclosure. The levels, predictions and thresholds above were fixed before any
perturbed network ran. The code was then smoke-tested on one run per
non-ideality (record 100, seed 0, `[128,128]`, 40 Hz, at 1 bit, sigma 1.5 and
50% stuck), and all three lost less than the model predicts. Nothing above
was changed after those runs; they are also rows of this grid.

Check: the exact-weight design points repeat B2's specs, so their ceilings must
match `results/h1_hardware_cost/b2_grid.csv` exactly.

Usage:
    uv run python scripts/h1_hardware_cost/pilot_a_predictions.py
    OMP_NUM_THREADS=1 uv run python scripts/h1_hardware_cost/pilot_a_grid.py
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
LAYER_WIDTH = 128
INPUT_WIDTHS: tuple[int, ...] = (128, 512, 2048)
ENCODER_RATES_HZ: tuple[float, ...] = (10.0, 20.0, 40.0)
BITS: tuple[int, ...] = (1, 2, 3, 4)
SIGMAS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
STUCK: tuple[float, ...] = (0.25, 0.5, 0.75)
# PA4's matched-PR pairs: (level with a narrow spread, level with a wide one).
MATCHED_PAIRS: tuple[tuple[str, str], ...] = (
    ("stuck 0.5", "sigma 1.5"),
    ("1 bit", "sigma 1"),
)
OUTPUT = Path("results/h1_hardware_cost/pilot_a_grid.csv")

# name -> (bits, sigma, stuck)
Level = tuple[int | None, float | None, float | None]


def levels() -> dict[str, Level]:
    table: dict[str, Level] = {"exact": (None, None, None)}
    table.update({f"{b} bit": (b, None, None) for b in BITS})
    table.update({f"sigma {s:g}": (None, s, None) for s in SIGMAS})
    table.update({f"stuck {p:g}": (None, None, p) for p in STUCK})
    return table


def level_of(spec: Spec) -> str:
    wanted = (spec.weight_bits, spec.weight_sigma, spec.weight_stuck)
    return next(name for name, level in levels().items() if level == wanted)


def specs(seeds: int, records: list[str]) -> list[Spec]:
    return [
        Spec(
            # Exact-weight points keep B2's experiment name so their keys, and
            # therefore their rows, are B2's.
            experiment="h1-b2" if name == "exact" else "h1-pilot-a",
            signal="mit-bih",
            unit=record,
            seed=seed,
            input_snr_db=CLEAN,
            layers=(n_in, LAYER_WIDTH),
            layer_rates_hz=(LAYER_RATE_HZ,),
            encoder_rate_hz=encoder_hz,
            fan_in=FAN_IN,
            weight_bits=bits,
            weight_sigma=sigma,
            weight_stuck=stuck,
        )
        for record in records
        for seed in range(seeds)
        for n_in in INPUT_WIDTHS
        for encoder_hz in ENCODER_RATES_HZ
        for name, (bits, sigma, stuck) in levels().items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    grid = specs(args.seeds, list(args.records))
    print(f"H1 pilot A: {len(grid)} runs -> {OUTPUT}")
    run_grid(grid, OUTPUT, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
