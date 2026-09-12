"""
M2 — locating the bottleneck.

M1a established that the ~1 dB output ceiling is not caused by the input
representation: giving the network genuinely independent information about the
clean signal (32x the input width, 15 dB of available gain) bought +0.25 dB.
So the information dies somewhere inside the pipeline. This script finds where.

Every stage is scored two ways, and the difference between them is the point:

  population readout   Exactly what the system does today — the
                       `PopulationDecoder` arithmetic (exponential smoothing,
                       rate conversion, clip to [0,1], affine map, mean across
                       neurons) applied to that stage's spikes.

  optimal linear       The best any *linear* decoder could do from that stage:
  readout              least-squares from the per-neuron smoothed traces onto
                       the clean signal. Fit on the first half, scored on the
                       second, so overfitting is visible rather than hidden.

Read the two columns together:

  - Both drop at the same stage  -> information is genuinely destroyed there.
  - Linear stays high, population -> the information survives; the decoder
    drops                           cannot reach it. A fixable engineering bug.

The clip column tests one specific suspect directly: `PopulationDecoder.decode`
clips normalised rate to [0, 1] (population.py:61), so any instantaneous rate
above `max_firing_rate_hz` is discarded outright.

The `--uniform` control runs [N, N, N] so layer width never changes between
stages, separating true transformation loss from the readout noise that comes
with simply having fewer neurons to average over.

Usage:
    uv run python scripts/m2_bottleneck/run_m2.py
    uv run python scripts/m2_bottleneck/run_m2.py --uniform --seeds 5
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
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, load_clean_ecg
from neuroflux_datasets.synthetic import add_gaussian_noise

FS = 250.0
DT = 1000.0 / FS
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0


@dataclass(frozen=True)
class StageResult:
    """One stage of the pipeline, scored two ways."""

    stage: str
    order: int
    n_units: int
    seed: int
    unit: str
    pop_snr_db: float
    linear_train_db: float
    linear_test_db: float
    clip_fraction: float


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
    traces: np.ndarray, v_min: float, v_max: float, tau: float = TAU_DECODE
) -> tuple[np.ndarray, float]:
    """
    Replicate `PopulationDecoder.decode` on precomputed traces.

    Returns the decoded scalar series and the fraction of (time, neuron)
    samples that the clip discarded.
    """
    rate_hz = traces / (tau / 1000.0)
    normalized = rate_hz / MAX_RATE_HZ
    clipped = np.clip(normalized, 0.0, 1.0)
    clip_fraction = float(np.mean(normalized > 1.0))
    decoded = v_min + clipped * (v_max - v_min)
    return decoded.mean(axis=1), clip_fraction


def linear_readout(features: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """
    Best linear estimate of `target` from `features`, split-half validated.

    Fits on the first half and scores on the second. Reporting both halves
    makes overfitting visible: a large train/test gap means the stage has more
    free parameters than the data supports, not more information.
    """
    n_steps = features.shape[0]
    split = n_steps // 2

    def design(block: np.ndarray) -> np.ndarray:
        return np.hstack([block, np.ones((block.shape[0], 1))])

    train_x, test_x = design(features[:split]), design(features[split:])
    train_y, test_y = target[:split], target[split:]

    weights, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return (
        affine_corrected_snr(train_y, train_x @ weights),
        affine_corrected_snr(test_y, test_x @ weights),
    )


def run_seed(
    seed: int,
    input_snr_db: float,
    duration_s: float,
    layers: list[int],
    epochs: int,
    tau_decode: float = TAU_DECODE,
    dataset: str = "synthetic",
    unit: str = "synthetic",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> list[StageResult]:
    """Run one seed and score every stage of the pipeline."""
    rng = np.random.default_rng(seed)
    clean = load_clean_ecg(
        dataset,
        unit,
        duration_s=duration_s,
        fs=FS,
        window=seed,
        cache_dir=cache_dir,
    )
    noisy = add_gaussian_noise(clean, input_snr_db, rng)
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
    calibrate_gain(net, calib_spikes, dt=DT, target_rate_hz=TARGET_RATE_HZ, max_iter=22)

    n_steps = drive.shape[0]

    if epochs:
        # Save and restore the encoder RNG around training, so the evaluation
        # pass sees an identical spike stream regardless of epoch count. Without
        # this, an epochs=0 vs epochs=N comparison confounds "what training did"
        # with "a different noise realisation". Mirrors cli.py:303-311.
        rng_state = encoder.rng.bit_generator.state
        for _ in range(epochs):
            net.reset_transient_state()
            for t in range(n_steps):
                net.step(encoder.encode(drive[t], DT), DT, training=True)
        encoder.rng.bit_generator.state = rng_state

    # Capture every stage in one pass, plasticity off. Transient reset only —
    # `reset_state` would restore initial weights and silently discard training.
    net.reset_transient_state()
    captured: list[list[np.ndarray]] = [[] for _ in range(len(layers))]
    for t in range(n_steps):
        all_spikes = net.step(encoder.encode(drive[t], DT), DT, training=False)
        for idx, spikes in enumerate(all_spikes):
            captured[idx].append(spikes)

    stage_names = ["encoded", *[f"layer{i + 1}" for i in range(len(layers) - 1)]]
    results: list[StageResult] = []

    # Stage 0: the analog input, before any spiking. The information available.
    results.append(
        StageResult(
            stage="analog_input",
            order=0,
            n_units=1,
            seed=seed,
            unit=unit,
            pop_snr_db=affine_corrected_snr(clean, noisy),
            linear_train_db=affine_corrected_snr(
                clean[: n_steps // 2], noisy[: n_steps // 2]
            ),
            linear_test_db=affine_corrected_snr(
                clean[n_steps // 2 :], noisy[n_steps // 2 :]
            ),
            clip_fraction=0.0,
        )
    )

    for idx, name in enumerate(stage_names):
        spikes = np.stack(captured[idx], axis=0)
        traces = smooth_traces(spikes, DT, tau_decode)
        decoded, clip_fraction = population_readout(traces, v_min, v_max, tau_decode)
        train_db, test_db = linear_readout(traces, clean)
        results.append(
            StageResult(
                stage=name,
                order=idx + 1,
                n_units=spikes.shape[1],
                seed=seed,
                unit=unit,
                pop_snr_db=affine_corrected_snr(clean, decoded),
                linear_train_db=train_db,
                linear_test_db=test_db,
                clip_fraction=clip_fraction,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--input-snr", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--tau", type=float, default=TAU_DECODE)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="explicit layer widths, e.g. --layers 128 128 128",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="use [32,32,32] so width never changes between stages",
    )
    parser.add_argument(
        "--dataset", choices=["synthetic", "mit-bih"], default="synthetic"
    )
    parser.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=None,
        help="MIT-BIH records; the record is the unit of analysis (default: 214)",
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("results/m2_bottleneck/m2_results.csv")
    )
    args = parser.parse_args()

    if args.layers:
        layers = list(args.layers)
    else:
        layers = [32, 32, 32] if args.uniform else [32, 16, 8]
    label = f"layers={layers}, tau={args.tau:.0f}ms"

    print(
        f"M2 — per-stage information localisation   ({label}, "
        f"epochs={args.epochs}, input SNR={args.input_snr} dB, {args.dataset}, "
        f"{args.seeds} seeds, {args.duration}s)"
    )
    print("All scores are affine-corrected SNR in dB.\n")

    units = (
        (list(args.records) if args.records else ["214"])
        if args.dataset == "mit-bih"
        else ["synthetic"]
    )
    results: list[StageResult] = []
    for unit in units:
        for seed in range(args.seeds):
            results.extend(
                run_seed(
                    seed,
                    args.input_snr,
                    args.duration,
                    layers,
                    args.epochs,
                    args.tau,
                    dataset=args.dataset,
                    unit=unit,
                    cache_dir=args.cache_dir,
                )
            )

    order = sorted({(r.order, r.stage, r.n_units) for r in results})
    print(
        f"{'stage':>14} {'units':>6} | {'population':>16} | {'linear (test)':>16} "
        f"| {'linear (train)':>14} | {'clipped':>8}"
    )
    print("-" * 92)
    prev_pop: float | None = None
    prev_lin: float | None = None
    for _, stage, n_units in order:
        rows = [r for r in results if r.stage == stage]
        pop = float(np.mean([r.pop_snr_db for r in rows]))
        pop_sd = float(np.std([r.pop_snr_db for r in rows]))
        lin = float(np.mean([r.linear_test_db for r in rows]))
        lin_sd = float(np.std([r.linear_test_db for r in rows]))
        train = float(np.mean([r.linear_train_db for r in rows]))
        clip = float(np.mean([r.clip_fraction for r in rows]))
        d_pop = "" if prev_pop is None else f"  ({pop - prev_pop:+.2f})"
        d_lin = "" if prev_lin is None else f"  ({lin - prev_lin:+.2f})"
        print(
            f"{stage:>14} {n_units:>6} | {pop:>7.2f} ±{pop_sd:>4.2f}{d_pop:>8} "
            f"| {lin:>7.2f} ±{lin_sd:>4.2f}{d_lin:>8} | {train:>14.2f} "
            f"| {clip * 100:>7.2f}%"
        )
        prev_pop, prev_lin = pop, lin

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[f.name for f in fields(StageResult)]
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
