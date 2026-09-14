"""
R3 C2, post hoc — why does training lower the ceiling at a held firing rate?

**Designed after C2's results were seen**, and reported separately from C2's
pre-registered analysis.

What C2 found. With the encoder and every layer recalibrated to 40 Hz after
training, STDP lowers the noiseless-input ceiling on MIT-BIH by 0.49, 0.60 and
0.64 dB after 10, 50 and 200 epochs, and by similar amounts on synthetic ECG.

The mechanism M3 offers. Each stage's intrinsic SNR is capped by the mean
pairwise correlation of its neurons' errors, which M3 traced to the cosine
similarity of their weight columns (3/4 for any non-negative random draw).
Every input channel carries the same tiled signal, so a Hebbian rule sees the
same pre-synaptic statistics at every post-synaptic neuron and should push
their weight columns toward each other.

Predictions, stated before this script was run:

  1. The mean weight-column cosine of each LIF layer rises with epochs.
  2. Across units, the fall in ceiling from 0 epochs tracks the rise in cosine:
     a negative Spearman correlation between the change in cosine and the
     change in ceiling, pooled over the three trained levels.
  3. If the cosine does not rise, M3's mechanism does not explain the loss, and
     the finding must say so.

The runs are C2's own training specs, so every ceiling here equals the value in
`c2_invariances.csv`; the script checks that and refuses to report otherwise.

Usage:
    uv run python scripts/r3_characterisation/c2_training_mechanism.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS
from scipy import stats  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_invariances import DECODER_TAUS, EPOCHS  # noqa: E402
from r3_common import CLEAN, Spec, run_spec  # noqa: E402

C2_CSV = Path("results/r3_characterisation/c2_invariances.csv")
DEFAULT_OUT = Path("results/r3_characterisation/c2_training_mechanism.csv")


@dataclass(frozen=True)
class Measurement:
    signal: str
    unit: str
    seed: int
    epochs: int
    ceiling_db: float
    verdict: str
    cosine_layer1: float
    cosine_layer2: float
    mean_weight_layer1: float
    mean_weight_layer2: float


def mean_column_cosine(weights: np.ndarray) -> float:
    """Mean pairwise cosine between weight columns (copied from M3's script)."""
    norms = np.linalg.norm(weights, axis=0)
    usable = norms > 0.0
    unit = weights[:, usable] / norms[usable]
    n = int(unit.shape[1])
    if n < 2:
        return float("nan")
    total = float(np.sum(np.square(np.sum(unit, axis=1))))
    return (total - n) / (n * (n - 1))


def measure(spec: Spec, cache_dir: str) -> Measurement:
    captured: list[np.ndarray] = []

    def hook(net: FeedForwardNetwork) -> None:
        captured.extend(np.asarray(s.weights, dtype=np.float64) for s in net.synapses)

    row = next(r for r in run_spec(spec, cache_dir, hook) if r.decoder_tau == 20.0)
    return Measurement(
        signal=spec.signal,
        unit=spec.unit,
        seed=spec.seed,
        epochs=spec.epochs,
        ceiling_db=row.snr_affine_db,
        verdict=row.verdict,
        cosine_layer1=mean_column_cosine(captured[0]),
        cosine_layer2=mean_column_cosine(captured[1]),
        mean_weight_layer1=float(captured[0].mean()),
        mean_weight_layer2=float(captured[1].mean()),
    )


def unit_of(r: Measurement) -> str:
    """Records are units on MIT-BIH; seeds are units on synthetic ECG."""
    return r.unit if r.signal == "mit-bih" else f"s{r.seed}"


def unit_mean(rows: list[Measurement], epochs: int, field: str, unit: str) -> float:
    vals = [
        getattr(r, field) for r in rows if r.epochs == epochs and unit_of(r) == unit
    ]
    return float(np.mean(vals)) if vals else math.nan


def specs(seeds: int, records: list[str]) -> list[Spec]:
    units = [("synthetic-ecg", "synthetic")] + [("mit-bih", r) for r in records]
    return [
        Spec(
            experiment="c2",
            signal=signal,
            unit=unit,
            seed=seed,
            input_snr_db=CLEAN,
            epochs=epochs,
            decoder_taus=DECODER_TAUS,
        )
        for signal, unit in units
        for seed in range(seeds)
        for epochs in EPOCHS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    grid = specs(args.seeds, list(args.records))
    print(f"R3 C2 training mechanism: {len(grid)} runs")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(measure, grid, [args.cache_dir] * len(grid)))

    # The runs must reproduce C2 exactly, or nothing below describes C2.
    # The committed copy is gzipped; a local grid run leaves the plain CSV.
    packed = C2_CSV.with_name(C2_CSV.name + ".gz")
    handle = (
        gzip.open(packed, "rt", newline="", encoding="utf-8")
        if not C2_CSV.exists() and packed.exists()
        else C2_CSV.open(newline="", encoding="utf-8")
    )
    with handle:
        stored = {
            r["spec_key"]: float(r["snr_affine_db"])
            for r in csv.DictReader(handle)
            if r["decoder_tau"] == "20.0"
        }
    for spec, result in zip(grid, results, strict=True):
        expected = stored.get(spec.key())
        if expected is None or not (
            (math.isnan(expected) and math.isnan(result.ceiling_db))
            or abs(expected - result.ceiling_db) < 1e-9
        ):
            raise SystemExit(f"Run does not reproduce C2: {spec.key()}")
    print("Every ceiling reproduces c2_invariances.csv exactly.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Measurement.__dataclass_fields__)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({f: getattr(r, f) for f in fields} for r in results)

    for signal in ("mit-bih", "synthetic-ecg"):
        block = [r for r in results if r.signal == signal and r.verdict == "ok"]
        units = sorted({unit_of(r) for r in block})

        print(f"{signal} (n = {len(units)} units)")
        print(
            f"  {'epochs':>6} {'ceiling':>8} {'cos L1':>7} {'cos L2':>7} "
            f"{'w L1':>6} {'w L2':>6}"
        )
        for epochs in EPOCHS:
            cols = [
                float(np.median([unit_mean(block, epochs, f, u) for u in units]))
                for f in (
                    "ceiling_db",
                    "cosine_layer1",
                    "cosine_layer2",
                    "mean_weight_layer1",
                    "mean_weight_layer2",
                )
            ]
            print(
                f"  {epochs:>6} {cols[0]:>8.2f} {cols[1]:>7.3f} {cols[2]:>7.3f} "
                f"{cols[3]:>6.3f} {cols[4]:>6.3f}"
            )
        for layer in ("cosine_layer1", "cosine_layer2"):
            d_cos, d_ceiling = [], []
            for u in units:
                for epochs in EPOCHS[1:]:
                    d_cos.append(
                        unit_mean(block, epochs, layer, u)
                        - unit_mean(block, 0, layer, u)
                    )
                    d_ceiling.append(
                        unit_mean(block, epochs, "ceiling_db", u)
                        - unit_mean(block, 0, "ceiling_db", u)
                    )
            pairs = [
                (a, b)
                for a, b in zip(d_cos, d_ceiling, strict=True)
                if math.isfinite(a + b)
            ]
            rho, p = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            print(
                f"  {layer}: Spearman(change in cosine, change in ceiling) "
                f"= {rho:+.3f} "
                f"(p = {p:.2g}, n = {len(pairs)} unit-level pairs)"
            )
        print()
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
