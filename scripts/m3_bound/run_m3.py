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

Usage:
    uv run python scripts/m3_bound/run_m3.py
    uv run python scripts/m3_bound/run_m3.py --layers 512 512 --tau 20
    uv run python scripts/m3_bound/run_m3.py --layers 128 128 128 --tau 4
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
    calibrate_gain,
)
from neuroflux_core.metrics.cascade import cascade_snr_db, intrinsic_snr_db
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_datasets.synthetic import add_gaussian_noise, generate_ecg

FS = 250.0
DT = 1000.0 / FS
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0

# The clean run is the measurement that anchors everything else, so it is
# marked with a sentinel rather than a very large SNR: `1/SNR_in` is exactly
# zero there, not merely small.
CLEAN = math.inf


@dataclass(frozen=True)
class StageRow:
    """One stage of one run, with everything the cascade model needs."""

    config: str
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
    traces: np.ndarray, v_min: float, v_max: float, tau: float
) -> np.ndarray:
    """Replicate `PopulationDecoder.decode` on precomputed traces."""
    rate_hz = traces / (tau / 1000.0)
    clipped = np.clip(rate_hz / MAX_RATE_HZ, 0.0, 1.0)
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
    seed: int,
    input_snr_db: float,
    duration_s: float,
    layers: list[int],
    tau_decode: float,
    target_rate_hz: float,
    config_label: str,
) -> list[StageRow]:
    """Run one seed at one input SNR and score every stage."""
    rng = np.random.default_rng(seed)
    clean = generate_ecg(duration_s=duration_s, fs=FS)
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
        max_firing_rate_hz=MAX_RATE_HZ,
        rng=np.random.default_rng(seed),
    )

    calib_steps = int(CALIBRATION_S * FS)
    calib_encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
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
        decoded = population_readout(traces, v_min, v_max, tau_decode)
        rows.append(
            StageRow(
                config=config_label,
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
    parser.add_argument("--layers", type=int, nargs="+", default=[32, 16, 8])
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
    stage_names = ["encoded", *[f"layer{i + 1}" for i in range(len(layers) - 1)]]

    print(f"M3 — cascade law   ({label}, {args.seeds} seeds, {args.duration}s)")
    print("All scores are affine-corrected SNR in dB.\n")

    all_rows: list[StageRow] = []

    # Step 1: the clean run. This is the entire model.
    clean_rows: list[StageRow] = []
    for seed in range(args.seeds):
        clean_rows.extend(
            run_seed(
                seed, CLEAN, args.duration, layers, args.tau, args.target_rate, label
            )
        )
    all_rows.extend(clean_rows)

    clean_snr = [mean_by_stage(clean_rows, name) for name in stage_names]
    lambdas: list[float] = []
    prev = math.inf
    print("Step 1 — clean input (SNR_in = inf). Per-stage intrinsic SNRs:\n")
    print(
        f"{'stage':>14} {'units':>6} {'SNR_out':>9} {'Lambda_k':>10} "
        f"{'rate Hz':>9} {'Fano':>7}"
    )
    print("-" * 62)
    for name, snr in zip(stage_names, clean_snr, strict=True):
        lam = snr if math.isinf(prev) else intrinsic_snr_db(prev, snr)
        lambdas.append(lam)
        stage_rows = [r for r in clean_rows if r.stage == name]
        print(
            f"{name:>14} {stage_rows[0].n_units:>6} {snr:>9.2f} {lam:>10.2f} "
            f"{np.mean([r.mean_rate_hz for r in stage_rows]):>9.1f} "
            f"{np.mean([r.fano for r in stage_rows]):>7.3f}"
        )
        prev = snr
    ceiling = clean_snr[-1]
    print(f"\n  Predicted ceiling (= clean-input output) = {ceiling:.2f} dB")

    # Step 2: predict every noisy run from the clean run alone.
    print("\nStep 2 — prediction vs measurement. No parameters are fitted here.\n")
    print(
        f"{'SNR_in':>8} {'stage':>14} {'predicted':>10} {'measured':>9} {'residual':>9}"
    )
    print("-" * 54)
    residuals: list[float] = []
    for nominal in args.input_snrs:
        rows: list[StageRow] = []
        for seed in range(args.seeds):
            rows.extend(
                run_seed(
                    seed,
                    nominal,
                    args.duration,
                    layers,
                    args.tau,
                    args.target_rate,
                    label,
                )
            )
        all_rows.extend(rows)
        measured_in = mean_by_stage(rows, "analog_input")
        for depth, name in enumerate(stage_names, start=1):
            predicted = cascade_snr_db(measured_in, lambdas[:depth])
            measured = mean_by_stage(rows, name)
            residuals.append(measured - predicted)
            print(
                f"{measured_in:>8.2f} {name:>14} {predicted:>10.2f} "
                f"{measured:>9.2f} {measured - predicted:>+9.2f}"
            )
        print("-" * 54)

    arr = np.asarray(residuals)
    print(
        f"\nResidual (measured - predicted) over {arr.size} stage-by-SNR points:\n"
        f"  mean {arr.mean():+.3f} dB   median {np.median(arr):+.3f} dB   "
        f"RMS {np.sqrt(np.mean(arr**2)):.3f} dB   max |.| {np.max(np.abs(arr)):.3f} dB"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(StageRow)])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
