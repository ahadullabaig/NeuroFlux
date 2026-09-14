"""
R5b — do the QRS conclusions depend on the detector, or on skipping its filter?

Every QRS F1 in R1, R2 and R3 comes from `neurokit2`'s default detector run on
the raw signal (`metrics/qrs.detect_rpeaks`). R1 §8 left the detector as an
untested dependency, and R3 §1.6 found that the same detector after neurokit's
standard `ecg_clean` recovers beats from the noisy input better than any
network did (roadmap next action 16).

**Runs.** Specs identical to two stored R3 grids, so each run must reproduce its
stored output SNR and F1 exactly (checked below, after the grid):

  rate      C3's output-rate axis on MIT-BIH: controlled `[32,16,8]`, hidden
            layer at 40 Hz, output at 0.25-80 Hz; noiseless, 0 dB, 10 dB.
  shipped   C1's shipped arms on MIT-BIH: passive and 10-epoch networks at
            Gaussian -5, 0, 5, 10 and 20 dB.

16 records x 10 seeds. The noisy input and the network output (population
decoder, tau = 20 ms) are each scored with three detectors — `neurokit`,
`pantompkins1985`, `hamilton2002` — on the raw signal and after `ecg_clean`
with the detector's own cleaning method: six variants.

**What a pilot showed before this grid was fixed** (4 runs): at a dense 40 Hz
output the raw `neurokit` detector finds no beats (F1 0.00) where the same
detector after `ecg_clean` scores 0.90 and 0.78; raw `pantompkins1985` and
`hamilton2002` do worse than raw `neurokit` on the noisy input. Both
conclusions tested below may therefore depend on the missing filter.

**Analysis, fixed before the grid was run.** Unit: a record; F1 is averaged
over every run, silent or not (R2 protocol). Wilcoxon signed-rank paired by
record, BCa interval on the median difference, Holm within each question.

  Q1  R3 C3: "beat detection needs a sparse output". At 10 dB input, F1 with
      the output at 1 Hz minus at 40 Hz, under each of the six variants (Holm
      across six). The conclusion survives a variant if the difference is
      positive and significant; it is detector-independent only if it survives
      all six.
  Q2  R2 §5 / R3 C1: "training raises F1 over the passive shipped network". At
      each of the five inputs, 10 epochs minus passive, under each variant
      (Holm across 30). It survives a variant if positive and significant at
      every input.
  Q3  "No network condition beats a standard filter". At every input and
      detector, the cleaned noisy input minus the best network condition, where
      the best is the (condition, raw or cleaned output) with the highest
      median F1 at that input and detector (Holm across all input x detector
      cells). The conclusion stands unless some cell has the network
      significantly higher.
  Q4  Described: how much cleaning the network's own output changes its F1.

Usage:
    uv run python scripts/r5_generality/qrs_detectors.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, PHASE3_RECORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
from c3_operating_region import OUTPUT_RATES, layer_targets  # noqa: E402
from r3_common import CLEAN, FS, Spec, read_rows, run_grid, run_spec  # noqa: E402

DETECTORS: tuple[str, ...] = ("neurokit", "pantompkins1985", "hamilton2002")
RATE_INPUTS: tuple[float, ...] = (CLEAN, 0.0, 10.0)
SHIPPED_INPUTS: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0)
SHIPPED_EPOCHS: tuple[int, ...] = (0, 10)
STORED = {
    "c3-rate": Path("results/r3_characterisation/c3_rate.csv.gz"),
    "c1": Path("results/r3_characterisation/c1_flatness.csv.gz"),
}
DEFAULT_OUT = Path("results/r5_generality/qrs_detectors.csv")


@dataclass(frozen=True)
class DetectorRow:
    spec_key: str
    experiment: str
    unit: str
    seed: int
    operating_point: str
    epochs: int
    input_snr_db: float
    layer_rate_targets_hz: str
    out_rate_hz: float
    verdict: str
    snr_affine_db: float
    qrs_f1_default: float
    source: str
    detector: str
    cleaning: str
    f1: float
    sensitivity: float
    ppv: float
    jitter_ms: float


def _clean(signal: np.ndarray, method: str) -> np.ndarray:
    import neurokit2 as nk  # type: ignore[import-untyped]  # noqa: PLC0415

    return np.asarray(nk.ecg_clean(signal, sampling_rate=int(FS), method=method))


def score_detectors(spec: Spec, cache_dir: str) -> list[DetectorRow]:
    """One stored spec, re-run and scored with every detector variant."""
    captured: dict[str, np.ndarray] = {}

    def hook(
        clean: np.ndarray,
        noisy: np.ndarray,
        beats: np.ndarray | None,
        outputs: dict[tuple[str, float], np.ndarray],
    ) -> None:
        if beats is None:
            raise ValueError("Detector scoring needs annotated beats.")
        captured["input"] = noisy
        captured["output"] = outputs[("population", 20.0)]
        captured["beats"] = beats

    (row,) = run_spec(spec, cache_dir, output_hook=hook)
    rows: list[DetectorRow] = []
    for source in ("input", "output"):
        signal = captured[source]
        for detector in DETECTORS:
            for cleaning in ("raw", "ecg_clean"):
                scored = signal if cleaning == "raw" else _clean(signal, detector)
                score = qrs_score(captured["beats"], scored, FS, method=detector)
                rows.append(
                    DetectorRow(
                        spec_key=spec.key(),
                        experiment=spec.experiment,
                        unit=spec.unit,
                        seed=spec.seed,
                        operating_point=spec.operating_point,
                        epochs=spec.epochs,
                        input_snr_db=spec.input_snr_db,
                        layer_rate_targets_hz=row.layer_rate_targets_hz,
                        out_rate_hz=row.out_rate_hz,
                        verdict=row.verdict,
                        snr_affine_db=row.snr_affine_db,
                        qrs_f1_default=row.qrs_f1,
                        source=source,
                        detector=detector,
                        cleaning=cleaning,
                        f1=score.f1,
                        sensitivity=score.sensitivity,
                        ppv=score.ppv,
                        jitter_ms=score.rms_jitter_ms,
                    )
                )
    return rows


def build_specs(seeds: int, records: list[str]) -> list[Spec]:
    specs: list[Spec] = []
    for record in records:
        for seed in range(seeds):
            for rate in OUTPUT_RATES:
                for snr in RATE_INPUTS:
                    specs.append(
                        Spec(
                            experiment="c3-rate",
                            signal="mit-bih",
                            unit=record,
                            seed=seed,
                            input_snr_db=snr,
                            layer_rates_hz=layer_targets("output", 2, rate),
                        )
                    )
            for epochs in SHIPPED_EPOCHS:
                for snr in SHIPPED_INPUTS:
                    specs.append(
                        Spec(
                            experiment="c1",
                            signal="mit-bih",
                            unit=record,
                            seed=seed,
                            input_snr_db=snr,
                            operating_point="shipped",
                            epochs=epochs,
                        )
                    )
    return specs


def _same(stored: str, new: str) -> bool:
    a, b = float(stored), float(new)
    return (math.isnan(a) and math.isnan(b)) or a == b


def verify(out: Path, specs: list[Spec]) -> None:
    """Every run's output SNR and default F1 must equal its stored R3 row."""
    wanted = {s.key() for s in specs}
    stored: dict[str, tuple[str, str]] = {}
    for path in STORED.values():
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["spec_key"] in wanted and r["decoder_tau"] == "20.0":
                    stored[r["spec_key"]] = (r["snr_affine_db"], r["qrs_f1"])
    missing = wanted - set(stored)
    if missing:
        raise SystemExit(f"{len(missing)} specs have no stored R3 row.")
    seen: set[str] = set()
    for r in read_rows(out):
        snr, f1 = stored[r["spec_key"]]
        if not (_same(snr, r["snr_affine_db"]) and _same(f1, r["qrs_f1_default"])):
            raise SystemExit(f"Run does not reproduce R3: {r['spec_key']}")
        if r["source"] == "output" and r["detector"] == "neurokit":
            if r["cleaning"] == "raw" and not _same(f1, r["f1"]):
                raise SystemExit(f"Default detector disagrees: {r['spec_key']}")
        seen.add(r["spec_key"])
    if seen != wanted:
        raise SystemExit(f"{len(wanted - seen)} specs missing from {out}.")
    print(f"All {len(seen)} runs reproduce their stored R3 rows exactly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--records", nargs="+", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = build_specs(args.seeds, list(args.records))
    print(f"R5b QRS detectors: {len(specs)} runs -> {args.out}")
    run_grid(
        specs,
        args.out,
        workers=args.workers,
        cache_dir=args.cache_dir,
        task=score_detectors,
        row_type=DetectorRow,
    )
    verify(args.out, specs)


if __name__ == "__main__":
    main()
