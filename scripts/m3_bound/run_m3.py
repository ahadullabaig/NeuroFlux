"""
M3 — the analytical bound, tested.

M2 localised the loss to the LIF layers but left the mechanism unstated, and
its "each layer retains ~22%" figure turned out to be a ratio of decibel
values rather than of powers (see `docs/m3_findings.md` §1). This script
tests the law that replaces it.

`metrics/cascade.py` derives that inverse SNRs add along a pipeline:

    1/SNR_out = 1/SNR_in + sum_k 1/Lambda_k

where `Lambda_k` is stage `k`'s intrinsic SNR — what it would deliver from a
noiseless input. The law's falsifiable content is that the `Lambda_k` are
properties of the *architecture*, independent of the input. So:

  1. Run the pipeline on a **noiseless** signal. With `1/SNR_in = 0` the
     measured per-stage SNRs give the `Lambda_k` directly.
  2. Predict the output at every other input SNR from those numbers alone.
  3. Compare against measurement. **There are no free parameters** — nothing
     is fitted to the noisy runs.

If the prediction holds, three things follow at once: the ceiling is exactly
the clean-signal output, depth is strictly harmful, and the invariance to
every architectural knob measured in Findings 1-5 is a consequence of which
stage dominates the sum rather than a separate fact about each knob.

Spike statistics (rate, Fano factor) are recorded alongside so the same runs
feed the structural model in `noise_structure.py`.

The unit of analysis is a **record**, per the R2 protocol: each record gets its
own `Lambda_k` from its own clean run, and its own prediction is checked against
its own measurement. Sixteen records therefore give sixteen independent tests of
the law rather than one pooled test.

Usage:
    uv run python scripts/m3_bound/run_m3.py
    uv run python scripts/m3_bound/run_m3.py --layers 512 512 --tau 20
    uv run python scripts/m3_bound/run_m3.py --dataset mit-bih --seeds 2
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from neuroflux_core import (
    LIFConfig,
    Network,
    PoissonRateEncoder,
    STDPConfig,
    calibrate_encoder_rate,
    calibrate_gain,
)
from neuroflux_core.metrics.cascade import cascade_snr_db, intrinsic_snr_db
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_datasets.signals import PHASE3_RECORDS, load_clean_ecg
from neuroflux_datasets.synthetic import add_gaussian_noise

FS = 250.0
DT = 1000.0 / FS
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0
CACHE_DIR = "data/mit_bih"

# The clean run is the measurement that anchors everything else, so it is
# marked with a sentinel rather than a very large SNR: `1/SNR_in` is exactly
# zero there, not merely small.
CLEAN = math.inf


@dataclass(frozen=True)
class StageRow:
    """One stage of one run, with everything the cascade model needs."""

    config: str
    unit: str
    input_snr_nominal: float
    input_snr_measured: float
    stage: str
    order: int
    n_units: int
    seed: int
    snr_db: float
    mean_rate_hz: float
    fano: float
    output_std: float
    degenerate: bool


# Duplicated from scripts/m2_bottleneck/run_m2.py rather than imported: the
# scripts tree is not an importable package, and mypy runs over it in strict
# mode. Keep the two copies in step.
def smooth_traces(spikes: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Per-neuron exponential spike traces, matching PopulationDecoder."""
    decay = math.exp(-dt / tau)
    traces = np.empty(spikes.shape, dtype=np.float64)
    state = np.zeros(spikes.shape[1], dtype=np.float64)
    for t in range(spikes.shape[0]):
        state = state * decay + spikes[t].astype(np.float64)
        traces[t] = state
    return traces


def population_readout(
    traces: np.ndarray,
    v_min: float,
    v_max: float,
    tau: float,
    max_rate_hz: float = MAX_RATE_HZ,
) -> np.ndarray:
    """Replicate `PopulationDecoder.decode` on precomputed traces."""
    rate_hz = traces / (tau / 1000.0)
    clipped = np.clip(rate_hz / max_rate_hz, 0.0, 1.0)
    decoded = v_min + clipped * (v_max - v_min)
    return np.asarray(decoded.mean(axis=1))


def fano_factor(spikes: np.ndarray, dt: float, window_ms: float) -> float:
    """
    Spike-count Fano factor at the readout timescale, averaged over neurons.

    A Poisson train gives 1.0; the refractory period pushes a driven LIF
    population below it, which is one of the few things in this pipeline that
    helps the estimate rather than hurting it.
    """
    per_window = max(1, int(round(window_ms / dt)))
    n_windows = spikes.shape[0] // per_window
    if n_windows < 2:
        return float("nan")
    counts = (
        spikes[: n_windows * per_window]
        .reshape(n_windows, per_window, spikes.shape[1])
        .sum(axis=1)
        .astype(np.float64)
    )
    mean = counts.mean(axis=0)
    var = counts.var(axis=0)
    active = mean > 0.0
    if not np.any(active):
        return float("nan")
    return float(np.mean(var[active] / mean[active]))


def run_seed(
    clean: np.ndarray,
    seed: int,
    input_snr_db: float,
    layers: list[int],
    tau_decode: float,
    target_rate_hz: float,
    config_label: str,
    unit: str,
    encoder_rate_hz: float | None = None,
) -> list[StageRow]:
    """
    Run one seed at one input SNR on one signal, and score every stage.

    `encoder_rate_hz` holds the encoder's mean firing rate at that value by
    choosing its ceiling for this drive, the way `calibrate_gain` holds each
    LIF layer. `None` keeps the fixed `MAX_RATE_HZ` ceiling, whose rate then
    follows the signal's amplitude distribution.
    """
    rng = np.random.default_rng(seed)
    noisy = (
        clean.copy()
        if math.isinf(input_snr_db)
        else add_gaussian_noise(clean, input_snr_db, rng)
    )
    n_input = layers[0]
    drive = np.tile(noisy, (n_input, 1)).T

    v_min, v_max = float(np.min(drive)), float(np.max(drive))
    if v_max <= v_min:
        v_max = v_min + 1.0

    encoder_max_hz = (
        MAX_RATE_HZ
        if encoder_rate_hz is None
        else calibrate_encoder_rate(
            drive, v_min, v_max, DT, target_rate_hz=encoder_rate_hz
        )
    )

    net = Network.feedforward(
        layers=layers,
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=20.0, tau_minus=20.0),
        seed=seed,
        init_scheme="scaled",
    )
    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=encoder_max_hz,
        rng=np.random.default_rng(seed),
    )

    calib_steps = int(CALIBRATION_S * FS)
    calib_encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=encoder_max_hz,
        rng=np.random.default_rng(seed + 10_000),
    )
    calib_spikes = np.stack(
        [calib_encoder.encode(drive[t], DT) for t in range(calib_steps)], axis=0
    )
    calibrate_gain(net, calib_spikes, dt=DT, target_rate_hz=target_rate_hz, max_iter=22)

    n_steps = drive.shape[0]
    net.reset_transient_state()
    captured: list[list[np.ndarray]] = [[] for _ in range(len(layers))]
    for t in range(n_steps):
        for idx, spikes in enumerate(net.step(encoder.encode(drive[t], DT), DT, False)):
            captured[idx].append(spikes)

    # The clean run has no input noise at all, so its input SNR is infinite
    # rather than merely large. Everything downstream is then purely intrinsic.
    measured_in = (
        CLEAN if math.isinf(input_snr_db) else affine_corrected_snr(clean, noisy)
    )

    rows = [
        StageRow(
            config=config_label,
            unit=unit,
            input_snr_nominal=input_snr_db,
            input_snr_measured=measured_in,
            stage="analog_input",
            order=0,
            n_units=1,
            seed=seed,
            snr_db=measured_in,
            mean_rate_hz=float("nan"),
            fano=float("nan"),
            output_std=float(np.std(noisy)),
            degenerate=False,
        )
    ]

    stage_names = ["encoded", *[f"layer{i + 1}" for i in range(len(layers) - 1)]]
    for idx, name in enumerate(stage_names):
        spikes = np.stack(captured[idx], axis=0)
        traces = smooth_traces(spikes, DT, tau_decode)
        # The encoder stage decodes against its own ceiling; with a calibrated
        # encoder that ceiling can exceed MAX_RATE_HZ and the readout would clip.
        readout_max_hz = encoder_max_hz if idx == 0 else MAX_RATE_HZ
        decoded = population_readout(
            traces, v_min, v_max, tau_decode, max_rate_hz=readout_max_hz
        )
        rows.append(
            StageRow(
                config=config_label,
                unit=unit,
                input_snr_nominal=input_snr_db,
                input_snr_measured=measured_in,
                stage=name,
                order=idx + 1,
                n_units=int(spikes.shape[1]),
                seed=seed,
                snr_db=affine_corrected_snr(clean, decoded),
                mean_rate_hz=float(np.mean(spikes) * 1000.0 / DT),
                fano=fano_factor(spikes, DT, tau_decode),
                output_std=float(np.std(decoded)),
                degenerate=is_degenerate(decoded),
            )
        )
    return rows


def mean_by_stage(rows: list[StageRow], stage: str) -> float:
    """Mean SNR across seeds for one stage."""
    return float(np.mean([r.snr_db for r in rows if r.stage == stage]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=TAU_DECODE)
    parser.add_argument("--target-rate", type=float, default=TARGET_RATE_HZ)
    parser.add_argument(
        "--encoder-rate",
        type=float,
        default=None,
        help="hold the encoder's mean rate at this value (Hz) instead of fixing "
        "its ceiling at 100 Hz; controls the cross-record encoder confound",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[32, 16, 8])
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "mit-bih"],
        default="synthetic",
        help="signal source; mit-bih makes the record the unit of analysis",
    )
    parser.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=None,
        help="MIT-BIH records to use "
        f"(default: the {len(PHASE3_RECORDS)} Phase 3 records)",
    )
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--input-snrs",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 20.0, 30.0, 40.0],
        help="noisy input SNRs to predict; the clean run is always included",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/m3_bound/m3_cascade.csv")
    )
    args = parser.parse_args()

    layers = list(args.layers)
    label = f"layers={layers}, tau={args.tau:.0f}ms, rate={args.target_rate:.0f}Hz"
    if args.encoder_rate is not None:
        label += f", encoder={args.encoder_rate:.0f}Hz"
    stage_names = ["encoded", *[f"layer{i + 1}" for i in range(len(layers) - 1)]]

    if args.dataset == "mit-bih":
        units = list(args.records) if args.records else list(PHASE3_RECORDS)
    else:
        units = ["synthetic"]

    print(
        f"M3 — cascade law   ({label}, {args.dataset}, {len(units)} unit(s) x "
        f"{args.seeds} seed(s), {args.duration}s)"
    )
    print("All scores are affine-corrected SNR in dB.\n")

    all_rows: list[StageRow] = []
    # Each unit is tested against its own Lambdas, so a 16-record run is 16
    # independent tests of the law rather than one pooled test.
    lambdas_by_unit: dict[str, list[float]] = {}
    clean_by_unit: dict[str, list[np.ndarray]] = {}
    skipped: list[str] = []

    print("Step 1 — clean input (SNR_in = inf). Per-stage intrinsic SNRs:\n")
    print(
        f"{'unit':>10} {'stage':>10} {'units':>6} {'SNR_out':>9} {'Lambda_k':>10} "
        f"{'rate Hz':>9} {'Fano':>7}"
    )
    print("-" * 68)

    for unit in units:
        try:
            signals = [
                load_clean_ecg(
                    args.dataset,
                    unit,
                    duration_s=args.duration,
                    fs=FS,
                    window=seed,
                    cache_dir=args.cache_dir,
                )
                for seed in range(args.seeds)
            ]
        except (ValueError, FileNotFoundError) as exc:
            print(f"{unit:>10}  skipped — {exc}")
            skipped.append(unit)
            continue
        clean_by_unit[unit] = signals

        clean_rows: list[StageRow] = []
        for seed, signal in enumerate(signals):
            clean_rows.extend(
                run_seed(
                    signal,
                    seed,
                    CLEAN,
                    layers,
                    args.tau,
                    args.target_rate,
                    label,
                    unit,
                    args.encoder_rate,
                )
            )
        all_rows.extend(clean_rows)

        lambdas: list[float] = []
        prev = math.inf
        for name in stage_names:
            snr = mean_by_stage(clean_rows, name)
            lam = snr if math.isinf(prev) else intrinsic_snr_db(prev, snr)
            lambdas.append(lam)
            block = [r for r in clean_rows if r.stage == name]
            print(
                f"{unit:>10} {name:>10} {block[0].n_units:>6} {snr:>9.2f} "
                f"{lam:>10.2f} {np.mean([r.mean_rate_hz for r in block]):>9.1f} "
                f"{np.mean([r.fano for r in block]):>7.3f}"
            )
            prev = snr
        lambdas_by_unit[unit] = lambdas

    if not lambdas_by_unit:
        print("\nNo usable units. Nothing to test.")
        return

    ceilings = [
        mean_by_stage(
            [r for r in all_rows if r.unit == unit and math.isinf(r.input_snr_nominal)],
            stage_names[-1],
        )
        for unit in lambdas_by_unit
    ]
    print(
        f"\n  Predicted ceiling (= clean-input output), n={len(ceilings)} unit(s): "
        f"{np.mean(ceilings):.2f} ± {np.std(ceilings):.2f} dB "
        f"[{np.min(ceilings):.2f}, {np.max(ceilings):.2f}]"
    )

    # Step 2: predict every noisy run from that unit's own clean run.
    print("\nStep 2 — prediction vs measurement. No parameters are fitted here.\n")
    print(
        f"{'SNR_in':>8} {'stage':>10} {'predicted':>10} {'measured':>9} "
        f"{'residual':>18}"
    )
    print("-" * 58)
    # (nominal input SNR, residual) so the high-SNR regime can be reported
    # separately — the law is an additive-noise model, and the encoder's clip
    # is expected to break it at low input SNR.
    residuals: list[tuple[float, float]] = []
    for nominal in args.input_snrs:
        per_stage: dict[str, list[tuple[float, float]]] = {n: [] for n in stage_names}
        measured_ins: list[float] = []
        for unit, lambdas in lambdas_by_unit.items():
            rows: list[StageRow] = []
            for seed, signal in enumerate(clean_by_unit[unit]):
                rows.extend(
                    run_seed(
                        signal,
                        seed,
                        nominal,
                        layers,
                        args.tau,
                        args.target_rate,
                        label,
                        unit,
                        args.encoder_rate,
                    )
                )
            all_rows.extend(rows)
            measured_in = mean_by_stage(rows, "analog_input")
            measured_ins.append(measured_in)
            for depth, name in enumerate(stage_names, start=1):
                predicted = cascade_snr_db(measured_in, lambdas[:depth])
                measured = mean_by_stage(rows, name)
                per_stage[name].append((predicted, measured))
                residuals.append((nominal, measured - predicted))

        for name in stage_names:
            pairs = per_stage[name]
            pred = np.mean([p for p, _ in pairs])
            meas = np.mean([m for _, m in pairs])
            resid = np.array([m - p for p, m in pairs])
            print(
                f"{np.mean(measured_ins):>8.2f} {name:>10} {pred:>10.2f} "
                f"{meas:>9.2f} {resid.mean():>+9.2f} ±{resid.std():>6.2f}"
            )
        print("-" * 58)

    def summarise(label: str, values: list[float]) -> None:
        if not values:
            return
        arr = np.asarray(values)
        print(
            f"  {label:<24} n={arr.size:<5} mean {arr.mean():+.3f}   "
            f"median {np.median(arr):+.3f}   RMS {np.sqrt(np.mean(arr**2)):.3f}   "
            f"max |.| {np.max(np.abs(arr)):.3f}"
        )

    print("\nResidual (measured - predicted), dB:\n")
    summarise("all input SNRs", [d for _, d in residuals])
    summarise("input SNR >= 10 dB", [d for snr, d in residuals if snr >= 10.0])
    summarise("input SNR < 10 dB", [d for snr, d in residuals if snr < 10.0])
    print(
        "\n  The law is an additive-noise model. The encoder's clip to [0, 1] acts\n"
        "  as a limiter, compressing noise excursions but not signal, so the model\n"
        "  is expected to under-predict at low input SNR — and that deviation is\n"
        "  the regime advantage phase3_findings.md reports."
    )
    if skipped:
        print(f"\nSkipped units: {', '.join(skipped)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(StageRow)])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
