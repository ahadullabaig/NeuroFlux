"""
R3 C2 — what the ceiling does not depend on, with equivalence tests.

Finding 2 reported that sixteen times the output neurons bought +0.16 dB, that
the decoder time constant had an optimum rather than a trend, and that STDP
moved the ceiling by fractions of a dB — three seeds, synthetic ECG, no rate
control, and "no significant change" read as "no change". This grid repeats
those knobs and two more under per-layer *and* encoder rate control, and asks
each question with a test that can return "equivalent".

**Grid.** Controlled operating point (scaled init, encoder and every LIF layer
held at 40 Hz, recalibrated after any training); synthetic ECG and the 16
Phase 3 MIT-BIH records; 10 seeds; a noiseless input (the ceiling itself) and
a 10 dB Gaussian input.

  knob            levels                        reference   network
  output width N  4, 8, 16, 32, 64              8           [32, 16, N]
  decoder tau     5, 10, 20, 40, 80, 160 ms     20 ms       [32, 16, 8]
  training        0, 10, 50, 200 epochs         0           [32, 16, 8]
  depth           2, 3, 4 layers                3           [32, 8] / [32, 16, 8]
                                                            / [32, 16, 16, 8]
  STDP tau        5, 10, 20, 40, 80 ms          20 ms       [32, 16, 8], 10 epochs

Training is also run at the shipped operating point, where learning is free to
move firing rates, so the two readings of "learning does not change the
ceiling" — at a held rate, and as deployed — can be told apart.

**Analysis, fixed before the grid was run.**

  * Unit: a record on MIT-BIH, a seed on synthetic ECG. A unit's SNR is its
    mean over seeds whose verdict is `ok`. Its QRS F1 is the mean over *every*
    seed: a silent output detects nothing, and that F1 of 0 is the result, not
    a missing value — dropping it would flatter exactly the networks that fail.
  * Each non-reference level is paired with the reference by unit. Wilcoxon
    signed-rank, Holm across the levels of one (knob, signal, input, operating
    point) family, BCa interval on the median difference, and TOST at the
    pre-registered ±0.5 dB (±0.05 for QRS F1, R2's margin).
  * A knob is **invariant over its tested range** only if every level is
    equivalent. It **matters** if any level differs significantly by at least
    the margin. Anything else is reported level by level as inconclusive.
  * Output width is also set against the prediction a capacity argument makes
    — independent neurons, so the output layer's intrinsic SNR scales with N —
    computed per unit from that unit's own reference-width stage SNRs, with
    no fitted parameters. The width model's shared-noise correlation `c` is
    then fitted per unit, as a description only. Both are computed in `S/N`
    terms, converted from affine SNR (`1 + S/N`; `metrics/cascade.py`),
    because C1 showed the law cannot be applied to affine values directly.
  * The cascade law predicts that depth can only lower the ceiling, so depth is
    expected to *matter*; it is not a candidate invariance.

Usage:
    uv run python scripts/r3_characterisation/c2_invariances.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CLEAN, Spec, run_grid  # noqa: E402

INPUTS: tuple[float, ...] = (CLEAN, 10.0)
OUTPUT_WIDTHS: tuple[int, ...] = (4, 8, 16, 32, 64)
DECODER_TAUS: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
EPOCHS: tuple[int, ...] = (0, 10, 50, 200)
DEPTHS: tuple[tuple[int, ...], ...] = ((32, 8), (32, 16, 8), (32, 16, 16, 8))
STDP_TAUS: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
STDP_EPOCHS = 10
DEFAULT_OUT = Path("results/r3_characterisation/c2_invariances.csv")


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    """Every run in C2. Runs shared between knobs appear once."""
    units = [("synthetic-ecg", "synthetic")] + [("mit-bih", r) for r in records]
    by_key: dict[str, Spec] = {}

    def add(spec: Spec) -> None:
        by_key.setdefault(spec.key(), spec)

    for signal, unit in units:
        for seed in range(seeds):
            for snr in INPUTS:

                def spec(
                    layers: tuple[int, ...] = (32, 16, 8),
                    operating_point: str = "controlled",
                    epochs: int = 0,
                    stdp_tau: float = 20.0,
                    taus: tuple[float, ...] = DECODER_TAUS,
                    signal: str = signal,
                    unit: str = unit,
                    seed: int = seed,
                    snr: float = snr,
                ) -> Spec:
                    return Spec(
                        experiment="c2",
                        signal=signal,
                        unit=unit,
                        seed=seed,
                        input_snr_db=snr,
                        layers=layers,
                        operating_point=operating_point,
                        epochs=epochs,
                        stdp_tau=stdp_tau,
                        decoder_taus=taus,
                    )

                # Every run is scored at every decoder tau, which costs readout
                # time only; the tau knob reads the reference network's rows.
                for width in OUTPUT_WIDTHS:
                    add(spec(layers=(32, 16, width)))
                for layers in DEPTHS:
                    add(spec(layers=layers))
                for operating_point in ("controlled", "shipped"):
                    for epochs in EPOCHS:
                        add(spec(operating_point=operating_point, epochs=epochs))
                for stdp_tau in STDP_TAUS:
                    add(spec(epochs=STDP_EPOCHS, stdp_tau=stdp_tau))
    return list(by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = build_specs(args.seeds, list(args.records))
    print(f"R3 C2 — invariances: {len(specs)} runs -> {args.out}")
    run_grid(specs, args.out, workers=args.workers, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
