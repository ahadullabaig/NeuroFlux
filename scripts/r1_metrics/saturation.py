"""
R1 — does the metric discriminate at all?

Phase 3's headline clinical claim is that the passive network "preserves
R-peaks perfectly", `rpeak_integrity = 1.000` across every input SNR. That
number is an artifact of the metric. `rpeak.py:54-62` sets each signal's peak
prominence floor to `0.5 * std` of *that signal*, so the detection threshold
rises with the noise it is meant to penalise, and the score is computed between
two runs of the same detector rather than against ground truth.

This script measures the size of the problem, on the 16 Phase 3 records, and
checks that the replacement does not have it.

The test is the **identity baseline**: score the unprocessed noisy signal. Any
metric that reports near-perfect preservation for doing nothing cannot support
a claim about the architecture, because the architecture is not what produced
the score. A metric that discriminates must separate

    identity baseline   (the noisy signal, untouched)
    passive network     (the untrained pipeline's reconstruction)

by more than the seed-to-seed spread, across the SNR range — the R1 acceptance
criterion. Both conditions are scored with the old metric and the new one side
by side, so the comparison is between metrics, not between runs.

The unit of analysis is the record, per `docs/r2_findings.md`: each record
contributes one mean per (condition, input SNR), and the spread reported is
across records.

Usage:
    uv run python scripts/r1_metrics/saturation.py
    uv run python scripts/r1_metrics/saturation.py --seeds 5 --out results/r1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from neuroflux_core import (
    LIFConfig,
    Network,
    PoissonRateEncoder,
    PopulationDecoder,
    STDPConfig,
    calibrate_gain,
)
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.distortion import max_cross_correlation, prd1, rmse
from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_core.metrics.rpeak import rpeak_integrity
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_datasets.analysis import holm_bonferroni
from neuroflux_datasets.signals import PHASE3_RECORDS, load_reference_beats
from neuroflux_datasets.synthetic import add_flicker, add_gaussian_noise

FS = 250.0
DT = 1000.0 / FS
DURATION_S = 10.0
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0
LAYERS = [32, 16, 8]

# The Phase 3 corruption, unchanged, so these numbers speak to its claims.
FLICKER_HZ = 50.0
FLICKER_AMPLITUDE = 0.5
INPUT_SNRS = [-5.0, 0.0, 5.0, 10.0, 20.0]

CACHE_DIR = "data/mit_bih"


@dataclass(frozen=True)
class Row:
    """One (record, seed, input SNR, condition) measurement."""

    record: str
    seed: int
    input_snr_db: float
    condition: str
    rpeak_integrity: float
    qrs_f1: float
    qrs_sensitivity: float
    qrs_ppv: float
    qrs_jitter_ms: float
    prd1_pct: float
    rmse: float
    max_xcorr: float
    snr_db: float
    affine_snr_db: float
    degenerate: bool


def reconstruct(noisy: np.ndarray, seed: int, calibrate: bool) -> np.ndarray:
    """
    Passive (untrained) pipeline output for one noisy signal.

    `calibrate=False` is the Phase 3 path exactly — default `g_syn`, no gain
    calibration — because that is the configuration §5.6.2's claim was measured
    in, and re-scoring a different network would not test that claim.
    `calibrate=True` adds M1b's per-layer gain calibration, which is what every
    mechanism experiment from M2 on uses.
    """
    drive = np.tile(noisy, (LAYERS[0], 1)).T
    v_min, v_max = float(np.min(drive)), float(np.max(drive))
    if v_max <= v_min:
        v_max = v_min + 1.0

    net = Network.feedforward(
        layers=LAYERS,
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=20.0, tau_minus=20.0),
        seed=seed,
        init_scheme="scaled",
    )
    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
        rng=np.random.default_rng(seed),
    )
    if calibrate:
        calib_encoder = PoissonRateEncoder(
            v_min=v_min,
            v_max=v_max,
            max_firing_rate_hz=MAX_RATE_HZ,
            rng=np.random.default_rng(seed + 10_000),
        )
        calib_steps = int(CALIBRATION_S * FS)
        calib_spikes = np.stack(
            [calib_encoder.encode(drive[t], DT) for t in range(calib_steps)], axis=0
        )
        calibrate_gain(
            net, calib_spikes, dt=DT, target_rate_hz=TARGET_RATE_HZ, max_iter=22
        )

    decoder = PopulationDecoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
        tau_decode=TAU_DECODE,
    )
    net.reset_transient_state()
    out = np.empty(drive.shape[0], dtype=np.float64)
    for t in range(drive.shape[0]):
        spikes = net.step(encoder.encode(drive[t], DT), DT, False)[-1]
        out[t] = float(decoder.decode(spikes, DT)[0])
    return out


def score(
    record: str,
    seed: int,
    input_snr_db: float,
    condition: str,
    clean: np.ndarray,
    beats: np.ndarray,
    estimate: np.ndarray,
) -> Row:
    """Score one estimate of `clean` under both the old metric and the new."""
    qrs = qrs_score(beats, estimate, FS)
    return Row(
        record=record,
        seed=seed,
        input_snr_db=input_snr_db,
        condition=condition,
        rpeak_integrity=rpeak_integrity(clean, estimate, FS),
        qrs_f1=qrs.f1,
        qrs_sensitivity=qrs.sensitivity,
        qrs_ppv=qrs.ppv,
        qrs_jitter_ms=qrs.rms_jitter_ms,
        prd1_pct=prd1(clean, estimate),
        rmse=rmse(clean, estimate),
        max_xcorr=max_cross_correlation(clean, estimate, max_lag_samples=int(0.2 * FS)),
        snr_db=calculate_snr(clean, estimate),
        affine_snr_db=affine_corrected_snr(clean, estimate),
        degenerate=is_degenerate(estimate),
    )


def collect(records: list[str], seeds: int, cache_dir: str) -> list[Row]:
    """Every (record, seed, SNR, condition) row."""
    rows: list[Row] = []
    for record in records:
        for seed in range(seeds):
            clean, beats = load_reference_beats(
                record,
                duration_s=DURATION_S,
                fs=FS,
                window=seed,
                cache_dir=cache_dir,
            )
            rows.append(score(record, seed, float("inf"), "clean", clean, beats, clean))
            for snr_db in INPUT_SNRS:
                rng = np.random.default_rng(seed)
                flickered = add_flicker(clean, FS, FLICKER_HZ, FLICKER_AMPLITUDE)
                noisy = add_gaussian_noise(flickered, snr_db, rng)
                rows.append(
                    score(record, seed, snr_db, "identity", clean, beats, noisy)
                )
                for condition, calibrate in (
                    ("passive", False),
                    ("passive_calibrated", True),
                ):
                    rows.append(
                        score(
                            record,
                            seed,
                            snr_db,
                            condition,
                            clean,
                            beats,
                            reconstruct(noisy, seed, calibrate),
                        )
                    )
            print(f"  {record} seed {seed} done", flush=True)
    return rows


def summarise(rows: list[Row], metric: str) -> None:
    """Print the identity/passive separation for one metric, per input SNR."""
    print(f"\n{metric}")
    print(
        f"  {'SNR_in':>7}  {'identity':>16}  {'passive':>16}  "
        f"{'passive+calib':>16}  {'separation':>10}"
    )
    for snr in INPUT_SNRS:
        cells: dict[str, tuple[float, float]] = {}
        for condition in ("identity", "passive", "passive_calibrated"):
            per_record = [
                float(np.mean([float(getattr(r, metric)) for r in group]))
                for group in _grouped(rows, condition, snr)
                if group
            ]
            per_record = [v for v in per_record if not np.isnan(v)]
            if not per_record:
                cells[condition] = (float("nan"), float("nan"))
                continue
            cells[condition] = (
                float(np.mean(per_record)),
                float(np.std(per_record, ddof=1)) if len(per_record) > 1 else 0.0,
            )
        identity, passive = cells["identity"], cells["passive"]
        calibrated = cells["passive_calibrated"]
        pooled = float(np.sqrt((identity[1] ** 2 + passive[1] ** 2) / 2.0))
        separation = (
            abs(identity[0] - passive[0]) / pooled if pooled > 0 else float("inf")
        )
        print(
            f"  {snr:7.1f}  {identity[0]:8.3f} ±{identity[1]:6.3f}  "
            f"{passive[0]:8.3f} ±{passive[1]:6.3f}  "
            f"{calibrated[0]:8.3f} ±{calibrated[1]:6.3f}  {separation:9.2f} sd"
        )


def _grouped(rows: list[Row], condition: str, snr: float) -> list[list[Row]]:
    """Rows for one (condition, SNR), split into one list per record."""
    by_record: dict[str, list[Row]] = {}
    for row in rows:
        if row.condition == condition and row.input_snr_db == snr:
            by_record.setdefault(row.record, []).append(row)
    return list(by_record.values())


def dynamic_range(rows: list[Row], metric: str) -> tuple[float, float, float]:
    """
    Span of a metric over the identity baseline, across the input-SNR range.

    This is the decisive test, and a stricter one than any two-condition
    separation. Scoring the *unprocessed* signal at -5 dB and at 20 dB is
    scoring two signals that differ by 25 dB of noise with nothing else
    changed. A metric that returns the same number for both is not measuring
    the signal; a clinical claim resting on it is a claim about the metric.
    """
    lows = [
        float(getattr(r, metric))
        for r in rows
        if r.condition == "identity" and r.input_snr_db == min(INPUT_SNRS)
    ]
    highs = [
        float(getattr(r, metric))
        for r in rows
        if r.condition == "identity" and r.input_snr_db == max(INPUT_SNRS)
    ]
    low = float(np.mean([v for v in lows if not np.isnan(v)]))
    high = float(np.mean([v for v in highs if not np.isnan(v)]))
    return low, high, abs(high - low)


def paired_advantage(
    rows: list[Row], metric: str, snr: float
) -> tuple[float, float, int]:
    """
    Passive minus identity, paired on record, by Wilcoxon signed-rank.

    Returns:
        `(median difference, p-value, n pairs)`. The p-value is uncorrected;
        the caller applies Holm across the input-SNR family, per the R2
        protocol — five simultaneous tests is exactly the situation that
        protocol exists for.
    """
    from scipy import stats  # type: ignore[import-untyped]  # noqa: PLC0415

    identity = {
        group[0].record: float(np.mean([float(getattr(r, metric)) for r in group]))
        for group in _grouped(rows, "identity", snr)
    }
    passive = {
        group[0].record: float(np.mean([float(getattr(r, metric)) for r in group]))
        for group in _grouped(rows, "passive", snr)
    }
    paired = [
        passive[record] - identity[record]
        for record in sorted(set(identity) & set(passive))
        if not np.isnan(passive[record] - identity[record])
    ]

    if len(paired) < 2 or all(difference == 0.0 for difference in paired):
        return float("nan"), float("nan"), len(paired)
    result = stats.wilcoxon(paired)
    return float(np.median(paired)), float(result.pvalue), len(paired)


def write_csv(rows: list[Row], path: Path) -> None:
    """Write every measurement, so the summary can be recomputed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Row)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {path}")


def read_csv(path: Path) -> list[Row]:
    """Reload a previous run, so the analysis can be redone without rerunning."""
    types = {f.name: f.type for f in fields(Row)}
    rows: list[Row] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            values: dict[str, object] = {}
            for name, text in raw.items():
                if types[name] == "str":
                    values[name] = text
                elif types[name] == "int":
                    values[name] = int(text)
                elif types[name] == "bool":
                    values[name] = text == "True"
                else:
                    values[name] = float(text)
            rows.append(Row(**values))  # type: ignore[arg-type]
    return rows


def main() -> None:
    """Run the saturation comparison and print the separation table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--records", nargs="*", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--out", default="results/r1_metrics/saturation.csv")
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Re-read --out and reprint the tables without rerunning anything.",
    )
    args = parser.parse_args()

    if args.analyse_only:
        rows = read_csv(Path(args.out))
        print(f"Re-analysing {len(rows)} rows from {args.out}")
    else:
        print(
            f"R1 saturation: {len(args.records)} records x {args.seeds} seeds x "
            f"{len(INPUT_SNRS)} input SNRs, {FLICKER_HZ:.0f} Hz flicker at "
            f"{FLICKER_AMPLITUDE}"
        )
        rows = collect(args.records, args.seeds, args.cache_dir)

    print("\n" + "=" * 72)
    print("Identity baseline vs passive network, record as the unit of analysis")
    print("Separation is |difference| in pooled across-record standard deviations.")
    print("=" * 72)
    for metric in ("rpeak_integrity", "qrs_f1", "qrs_sensitivity", "qrs_ppv"):
        summarise(rows, metric)
    for metric in ("prd1_pct", "max_xcorr", "affine_snr_db"):
        summarise(rows, metric)

    print("\n" + "=" * 72)
    print("Dynamic range over the identity baseline, -5 dB to 20 dB input")
    print("Nothing but the noise changes between these two columns.")
    print("=" * 72)
    print(f"  {'metric':>18}  {'at -5 dB':>10}  {'at 20 dB':>10}  {'span':>10}")
    for metric in (
        "rpeak_integrity",
        "qrs_f1",
        "prd1_pct",
        "max_xcorr",
        "affine_snr_db",
    ):
        low, high, span = dynamic_range(rows, metric)
        print(f"  {metric:>18}  {low:10.3f}  {high:10.3f}  {span:10.3f}")

    print("\n" + "=" * 72)
    print("Passive network minus identity baseline, paired on record (qrs_f1)")
    print("Holm-Bonferroni across the five input SNRs, per the R2 protocol.")
    print("=" * 72)
    results = [paired_advantage(rows, "qrs_f1", snr) for snr in INPUT_SNRS]
    adjusted = holm_bonferroni([1.0 if np.isnan(p) else p for _, p, _ in results])
    print(f"  {'SNR_in':>7}  {'n':>3}  {'median':>8}  {'p':>8}  {'p_holm':>8}  verdict")
    for snr, (median, p_value, n), p_holm in zip(
        INPUT_SNRS, results, adjusted, strict=True
    ):
        verdict = (
            "advantage"
            if p_holm < 0.05 and median > 0
            else ("cost" if p_holm < 0.05 else "no effect")
        )
        print(
            f"  {snr:7.1f}  {n:3d}  {median:+8.3f}  {p_value:8.4f}  "
            f"{p_holm:8.4f}  {verdict}"
        )

    clean_rows = [r for r in rows if r.condition == "clean"]
    if clean_rows:
        clean_f1 = float(np.mean([r.qrs_f1 for r in clean_rows]))
        print(
            f"\nDetector ceiling on the clean signal: mean F1 {clean_f1:.3f} "
            f"over {len(clean_rows)} runs. Any score at or near this is "
            f"indistinguishable from no degradation."
        )

    if not args.analyse_only:
        write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
