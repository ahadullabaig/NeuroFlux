"""
R4b (roadmap next action 17) — does training collapse layer 1's effective fan-in?

R3 C2 §2.6 found that at a held 40 Hz rate STDP lowers the ceiling mostly in
layer 1, whose weight columns *decorrelate* (cosine 0.76 -> 0.37 on MIT-BIH) —
the opposite of what M3's column-cosine mechanism needs to explain a loss. It
also found layer-1 weights shrinking fourfold (mean 0.15 -> 0.04). The
hypothesis recorded there: training leaves each layer-1 neuron driven by a few
surviving synapses, recalibration raises the gain to hold 40 Hz, and M4's
finding that too few effective inputs per neuron raises its noise does the rest.

The effective number of inputs to neuron j is the participation ratio of its
weight column, `PR_j = (sum_i w_ij)^2 / sum_i w_ij^2`: N for equal weights, 1
for a single non-zero weight.

**Predictions, stated before this script was run.**

  P1  Layer-1 mean participation ratio falls with training (median over units,
      0 -> 10 -> 50 -> 200 epochs).
  P2  Across units and the three trained levels, the change in layer-1 PR from
      0 epochs correlates *positively* with the change in layer-1 intrinsic
      `S/N` (Spearman, p < 0.05), on MIT-BIH and on synthetic ECG.
  P3  The PR account beats the cosine account for layer 1: |rho(PR)| exceeds
      |rho(cosine)| on both signals. If P2 holds but P3 does not, fan-in and
      decorrelation are confounded here and neither is established.

The runs are R3 C2's own training specs (rate-controlled, `[32,16,8]`,
noiseless input, 10 seeds, 16 MIT-BIH records + synthetic ECG), so every
ceiling must equal `results/r3_characterisation/c2_invariances.csv.gz`; the
script refuses to write otherwise. Intrinsic `S/N` per stage comes from each
run's recorded stage SNRs, converted from affine SNR, as in R3 §2.6.

Usage:
    uv run python scripts/r4b_mechanism/training_fan_in.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from c2_invariances import DECODER_TAUS, EPOCHS  # noqa: E402
from r3_common import CLEAN, Spec, run_spec, split_floats  # noqa: E402

C2_DATA = Path("results/r3_characterisation/c2_invariances.csv.gz")
DEFAULT_OUT = Path("results/r4b_mechanism/training_fan_in.csv")


@dataclass(frozen=True)
class WeightRow:
    signal: str
    unit: str
    seed: int
    epochs: int
    verdict: str
    ceiling_db: float
    stage_affine_db: str
    cosine_layer1: float
    cosine_layer2: float
    participation_layer1: float
    participation_layer2: float
    mean_weight_layer1: float
    mean_weight_layer2: float


def mean_column_cosine(weights: np.ndarray) -> float:
    """Mean pairwise cosine between weight columns (M3's formula)."""
    norms = np.linalg.norm(weights, axis=0)
    usable = norms > 0.0
    unit = weights[:, usable] / norms[usable]
    n = int(unit.shape[1])
    if n < 2:
        return math.nan
    total = float(np.sum(np.square(np.sum(unit, axis=1))))
    return (total - n) / (n * (n - 1))


def mean_participation(weights: np.ndarray) -> float:
    """Mean participation ratio of the columns: the effective inputs per neuron."""
    sums = weights.sum(axis=0)
    squares = np.square(weights).sum(axis=0)
    usable = squares > 0.0
    if not np.any(usable):
        return 0.0
    return float(np.mean(np.square(sums[usable]) / squares[usable]))


def measure(spec: Spec, cache_dir: str) -> WeightRow:
    captured: list[np.ndarray] = []

    def hook(net: FeedForwardNetwork) -> None:
        captured.extend(np.asarray(s.weights, dtype=np.float64) for s in net.synapses)

    row = next(r for r in run_spec(spec, cache_dir, hook) if r.decoder_tau == 20.0)
    return WeightRow(
        signal=spec.signal,
        unit=spec.unit,
        seed=spec.seed,
        epochs=spec.epochs,
        verdict=row.verdict,
        ceiling_db=row.snr_affine_db,
        stage_affine_db=row.stage_affine_db,
        cosine_layer1=mean_column_cosine(captured[0]),
        cosine_layer2=mean_column_cosine(captured[1]),
        participation_layer1=mean_participation(captured[0]),
        participation_layer2=mean_participation(captured[1]),
        mean_weight_layer1=float(captured[0].mean()),
        mean_weight_layer2=float(captured[1].mean()),
    )


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
    print(f"R4b training fan-in: {len(grid)} runs")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(measure, grid, [args.cache_dir] * len(grid)))

    with gzip.open(C2_DATA, "rt", newline="", encoding="utf-8") as handle:
        stored = {
            r["spec_key"]: (
                float(r["snr_affine_db"]),
                split_floats(r["stage_affine_db"]),
            )
            for r in csv.DictReader(handle)
            if r["decoder_tau"] == "20.0"
        }
    for spec, result in zip(grid, results, strict=True):
        ceiling, stages = stored[spec.key()]
        same = (math.isnan(ceiling) and math.isnan(result.ceiling_db)) or abs(
            ceiling - result.ceiling_db
        ) < 1e-9
        if not same or split_floats(result.stage_affine_db) != stages:
            raise SystemExit(f"Run does not reproduce R3 C2: {spec.key()}")
    print("Every run reproduces R3 C2 exactly.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(WeightRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
