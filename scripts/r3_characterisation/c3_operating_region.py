"""
R3 C3 — where the network works, and what its output rate does.

Two maps, both on the 16 Phase 3 MIT-BIH records with 10 seeds.

**(a) The gain plane.** Finding 3 showed that one global `g_syn` cannot serve
every input width: at `N_in = 64` the shipped gain saturates the network and
the naive `1/N_in` rescaling silences it. This grid maps the whole
`(N_in, g_syn)` plane of the shipped operating point — uniform init, one gain
for every layer, the encoder's ceiling at 100 Hz — so the working region can
be drawn rather than inferred from two points.

  network     [N_in, 16, 8],  N_in in 8, 16, 32, 64, 128, 256
  gain        25 values log-spaced from 1 to 300
  training    none, and 10 STDP epochs
  input       10 dB Gaussian

**(b) The output-rate axis.** R1 §5 found landmark recoverability only in
sparse outputs — a trained or scaled-init network firing at 0.2–5 Hz — and
every ceiling number so far was measured with the output held at 40 Hz. This
grid holds the encoder at 40 Hz and moves the output rate, at the controlled
operating point with no training.

  rate          0.25, 0.5, 1, 2, 5, 10, 20, 40, 80 Hz
  scope         output — hidden layer held at 40 Hz, output layer at the rate
                all    — every LIF layer at the rate
  network       [32, 16, 8]; and [32, 16, 64] at scope `output`, to ask
                whether width matters once the output is sparse
  signals       MIT-BIH and synthetic ECG
  input         noiseless (the ceiling), 0 dB and 10 dB Gaussian

**Analysis, fixed before the grids were run.**

  (a) A cell's verdict fractions are counted over all its runs. For each
      `N_in` and training arm, the **silent boundary** is the gain at which the
      silent fraction first falls to 0.5 or below, and the **saturated
      boundary** the gain at which the saturated fraction first reaches 0.5,
      both interpolated linearly in log gain between grid points. The usable
      band is the interval between them, in octaves; a silent boundary at or
      above the saturated one means no global gain works at that width.
      Affine output SNR is the median over records of each record's mean over
      `ok` seeds; QRS F1 averages every seed, since a silent output's F1 of 0
      is a result rather than a missing value.
  (b) The sparse-output hypothesis is that QRS F1 at 10 dB input is higher with
      the output at 2 Hz or less than at 40 Hz. Every rate is paired with 40 Hz
      by record: Wilcoxon, Holm across rates within (scope, network, input),
      TOST at ±0.05 F1. The ceiling and the output SNR get the same test at
      ±0.5 dB, which asks whether the ceiling measured at 40 Hz still describes
      the sparse regime. The cascade law is checked at every rate from that
      rate's own noiseless-input run, in `S/N` terms as C1 found it must be.
      Averaging follows (a): SNR over `ok` seeds, F1 over every seed.

**(c) Post hoc — the rate axis on non-ECG signals.** Added after (b)'s results
were seen: a sparse output behaves like a detector, which suits a spiky signal,
so (b) alone cannot say whether output rate matters for reconstruction in
general. Same arm as (b)'s `output [32, 16, 8]`, on synthetic EEG and the 2 Hz
sine, noiseless and 10 dB input, 10 seeds. Reported separately from (b).

Usage:
    uv run python scripts/r3_characterisation/c3_operating_region.py --part plane
    uv run python scripts/r3_characterisation/c3_operating_region.py --part rate
    uv run python scripts/r3_characterisation/c3_operating_region.py --part rate-signals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CLEAN, DENSE_RATE_HZ, Spec, run_grid  # noqa: E402

INPUT_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
GAINS: tuple[float, ...] = tuple(float(g) for g in np.geomspace(1.0, 300.0, 25))
PLANE_EPOCHS: tuple[int, ...] = (0, 10)
PLANE_INPUT_DB = 10.0

OUTPUT_RATES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
RATE_INPUTS: tuple[float, ...] = (CLEAN, 0.0, 10.0)
# (scope, network)
RATE_ARMS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("output", (32, 16, 8)),
    ("all", (32, 16, 8)),
    ("output", (32, 16, 64)),
)
OUTPUT_DIR = Path("results/r3_characterisation")


def plane_specs(seeds: int, records: list[str]) -> list[Spec]:
    """The shipped-operating-point gain plane."""
    return [
        Spec(
            experiment="c3-plane",
            signal="mit-bih",
            unit=record,
            seed=seed,
            input_snr_db=PLANE_INPUT_DB,
            layers=(width, 16, 8),
            operating_point="shipped",
            epochs=epochs,
            g_syn=gain,
        )
        for record in records
        for seed in range(seeds)
        for width in INPUT_WIDTHS
        for gain in GAINS
        for epochs in PLANE_EPOCHS
    ]


def layer_targets(scope: str, n_groups: int, rate: float) -> tuple[float, ...]:
    """Per-LIF-layer rate targets for one scope."""
    if scope == "all":
        return (rate,) * n_groups
    if scope == "output":
        return (DENSE_RATE_HZ,) * (n_groups - 1) + (rate,)
    raise ValueError(f"Unknown scope {scope!r}")


def rate_specs(seeds: int, records: list[str]) -> list[Spec]:
    """The output-rate axis at the controlled operating point."""
    units = [("synthetic-ecg", "synthetic")] + [("mit-bih", r) for r in records]
    by_key: dict[str, Spec] = {}
    for signal, unit in units:
        for seed in range(seeds):
            for scope, layers in RATE_ARMS:
                for rate in OUTPUT_RATES:
                    for snr in RATE_INPUTS:
                        spec = Spec(
                            experiment="c3-rate",
                            signal=signal,
                            unit=unit,
                            seed=seed,
                            input_snr_db=snr,
                            layers=layers,
                            layer_rates_hz=layer_targets(scope, len(layers) - 1, rate),
                        )
                        # At 40 Hz both scopes are the same network.
                        by_key.setdefault(spec.key(), spec)
    return list(by_key.values())


def rate_signal_specs(seeds: int) -> list[Spec]:
    """Post hoc: (b)'s output-only arm on EEG and the sine."""
    return [
        Spec(
            experiment="c3-rate-signals",
            signal=signal,
            unit="synthetic",
            seed=seed,
            input_snr_db=snr,
            layers=(32, 16, 8),
            layer_rates_hz=layer_targets("output", 2, rate),
        )
        for signal in ("eeg", "sine")
        for seed in range(seeds)
        for rate in OUTPUT_RATES
        for snr in (CLEAN, 10.0)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=["plane", "rate", "rate-signals"], required=True
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = list(args.records)
    if args.part == "plane":
        specs = plane_specs(args.seeds, records)
        out = args.out or OUTPUT_DIR / "c3_plane.csv"
    elif args.part == "rate":
        specs = rate_specs(args.seeds, records)
        out = args.out or OUTPUT_DIR / "c3_rate.csv"
    else:
        specs = rate_signal_specs(args.seeds)
        out = args.out or OUTPUT_DIR / "c3_rate_signals.csv"
    print(f"R3 C3 — {args.part}: {len(specs)} runs -> {out}")
    run_grid(specs, out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
