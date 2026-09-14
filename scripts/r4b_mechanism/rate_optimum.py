"""
R4b (roadmap next action 13) — does the mechanism hold at the rate optimum?

Every mechanism number before R3 — M1a's width slope, M3's width model, M4's
fan-in intervention — was measured with every layer at 40 Hz. R3 C3 found the
MIT-BIH ceiling peaks with the output near 5 Hz, and that output width matters
more there (+0.3 dB from 8 to 64 neurons, against +0.07 at 40 Hz). Two parts,
both rate-controlled and untrained, both on the 16 MIT-BIH records with 10
seeds and a noiseless input.

**(width) Input width at a sparse output.** `[N_in, 16, 8]` for N_in in 8 to
256, output held at 5 Hz or 40 Hz, hidden layer and encoder at 40 Hz. The input
is tiled, as in M1a's `tiled` mode.

  Prediction: the per-record slope of ceiling on log2(N_in) is larger with the
  output at 5 Hz than at 40 Hz. Test: Wilcoxon signed-rank on the per-record
  slope difference; the slopes' medians get BCa intervals.

**(fan-in) M4's intervention against output rate.** `[N, N]` — one LIF layer, as
in M4 — at N = 128 with fan-in 8 and N = 512 with fan-in 16 (M4's optima), each
against its dense twin, with the layer held at 1, 2, 5, 10, 20 or 40 Hz.

  Prediction: the fan-in gain survives at the rate optimum — at the output rate
  where each network's ceiling is highest, sparse fan-in minus dense is
  significant and at least +0.5 dB, paired by record (Wilcoxon, Holm across
  the two widths; TOST ±0.5 dB reported). At 40 Hz the gain should reproduce
  M4's sign (M4: +0.73 dB at N = 128, +1.20 at 512, with an uncontrolled
  encoder and 8 s windows, so the size need not match).

Usage:
    uv run python scripts/r4b_mechanism/rate_optimum.py --part width
    uv run python scripts/r4b_mechanism/rate_optimum.py --part fan-in
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from r3_common import CLEAN, DENSE_RATE_HZ, Spec, run_grid  # noqa: E402

INPUT_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
WIDTH_OUTPUT_RATES: tuple[float, ...] = (5.0, 40.0)
FAN_IN_NETWORKS: tuple[tuple[int, int], ...] = ((128, 8), (512, 16))
FAN_IN_RATES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
OUTPUT_DIR = Path("results/r4b_mechanism")


def width_specs(seeds: int, records: list[str]) -> list[Spec]:
    return [
        Spec(
            experiment="r4b-width",
            signal="mit-bih",
            unit=record,
            seed=seed,
            input_snr_db=CLEAN,
            layers=(width, 16, 8),
            layer_rates_hz=(DENSE_RATE_HZ, rate),
        )
        for record in records
        for seed in range(seeds)
        for width in INPUT_WIDTHS
        for rate in WIDTH_OUTPUT_RATES
    ]


def fan_in_specs(seeds: int, records: list[str]) -> list[Spec]:
    return [
        Spec(
            experiment="r4b-fan-in",
            signal="mit-bih",
            unit=record,
            seed=seed,
            input_snr_db=CLEAN,
            layers=(width, width),
            layer_rates_hz=(rate,),
            fan_in=fan_in,
        )
        for record in records
        for seed in range(seeds)
        for width, k in FAN_IN_NETWORKS
        for fan_in in (None, k)
        for rate in FAN_IN_RATES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["width", "fan-in"], required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    records = list(args.records)
    if args.part == "width":
        specs, out = width_specs(args.seeds, records), OUTPUT_DIR / "width.csv"
    else:
        specs, out = fan_in_specs(args.seeds, records), OUTPUT_DIR / "fan_in.csv"
    print(f"R4b rate optimum, {args.part}: {len(specs)} runs -> {out}")
    run_grid(specs, out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
