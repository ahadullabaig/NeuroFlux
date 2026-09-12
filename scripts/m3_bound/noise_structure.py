"""
M3 — where `Lambda` comes from, and why width cannot lift it.

`run_m3.py` establishes that inverse SNRs add and that each stage's intrinsic
SNR `Lambda_k` predicts everything else. This script opens up `Lambda` itself.

Decompose a stage's population readout into the part a linear decoder can
predict from the clean signal and the part it cannot:

    trace_i(t) = a_i * clean(t) + b_i + r_i(t)

`r_i` is neuron `i`'s error. Averaging N neurons shrinks the *independent*
part of that error but not the part they share, so the population-mean error
variance is

    var_mean = var_single * (1/N + c * (1 - 1/N))

with `c` the mean pairwise correlation between neurons' errors. Hence

    Lambda(N) = var_signal / (var_single * (1/N + c * (1 - 1/N)))

and, taking N to infinity,

    Lambda(inf) = var_signal / (c * var_single).

**A single number, `c`, caps what any width can buy.** For c = 0.1 the most a
population can ever deliver is 10x the single-neuron SNR, no matter how many
neurons it contains — and 90% of that is reached by N = 90.

This is Moreno-Bote et al. (2014) information-limiting correlations, but
arising *inside* the network rather than at its input. M1a ruled out the input
as the source: giving the network genuinely independent input channels bought
+0.056 dB per doubling instead of the ideal +3.01. The neurons of a LIF layer
nevertheless share their drive, so they manufacture their own correlated noise
regardless of what arrives.

Two things are measured that the model does not fit:

  1. **The width law.** `c` is measured at one width and used to predict
     `Lambda` at every other width. The sweep tests that prediction.
  2. **What the shared error is made of.** Re-running the same network on the
     same analog drive with a different encoder RNG separates error that
     repeats (deterministic distortion — the LIF transfer nonlinearity) from
     error that does not (shared spiking noise). Only the second kind would
     ever average away.

Usage:
    uv run python scripts/m3_bound/noise_structure.py
    uv run python scripts/m3_bound/noise_structure.py --widths 16 32 64 128 256
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
from neuroflux_core.metrics.cascade import linear_to_db
from neuroflux_datasets.synthetic import generate_ecg

FS = 250.0
DT = 1000.0 / FS
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0


@dataclass(frozen=True)
class NoiseRow:
    """Error structure of one stage at one width."""

    width: int
    stage: str
    order: int
    seed: int
    n_units: int
    var_signal: float
    var_single: float
    var_mean: float
    noise_corr: float
    lambda_meas_db: float
    lambda_indep_db: float
    lambda_ceiling_db: float
    repeatable_fraction: float


def smooth_traces(spikes: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Per-neuron exponential spike traces, matching PopulationDecoder."""
    decay = math.exp(-dt / tau)
    traces = np.empty(spikes.shape, dtype=np.float64)
    state = np.zeros(spikes.shape[1], dtype=np.float64)
    for t in range(spikes.shape[0]):
        state = state * decay + spikes[t].astype(np.float64)
        traces[t] = state
    return traces


def residuals(traces: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split each neuron's trace into its linear signal component and its error.

    Returns:
        `(signal, error)`, both shaped like `traces`. The split is per neuron
        by least squares onto `[clean, 1]`, so `error` is exactly what no
        linear decoder could remove from that neuron.
    """
    design = np.vstack([clean, np.ones_like(clean)]).T
    coeffs, *_ = np.linalg.lstsq(design, traces, rcond=None)
    signal = design @ coeffs
    return signal, traces - signal


def noise_correlation(error: np.ndarray) -> tuple[float, float, float]:
    """
    Mean pairwise error correlation, without forming the N x N matrix.

    `var(mean) = (1/N^2) * (N * var_single + N(N-1) * c * var_single)` inverts
    to `c = (N * var_mean / var_single - 1) / (N - 1)`.

    Returns:
        `(var_single, var_mean, c)`.
    """
    n_units = error.shape[1]
    var_single = float(np.mean(np.var(error, axis=0)))
    var_mean = float(np.var(error.mean(axis=1)))
    if n_units < 2 or var_single <= 0.0:
        return var_single, var_mean, float("nan")
    corr = (n_units * var_mean / var_single - 1.0) / (n_units - 1)
    return var_single, var_mean, float(corr)


def lambda_at_width(
    var_signal: float, var_single: float, corr: float, n: float
) -> float:
    """`Lambda` predicted at width `n` from the error decomposition, in dB."""
    if var_single <= 0.0:
        return math.inf
    share = 1.0 / n if math.isfinite(n) else 0.0
    denom = var_single * (share + corr * (1.0 - share))
    if denom <= 0.0:
        return math.inf
    return linear_to_db(var_signal / denom)


def capture(
    width: int,
    seed: int,
    duration_s: float,
    tau_decode: float,
    target_rate_hz: float,
    encoder_seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run `[width, width]` on a clean ECG and return the clean signal and traces.

    The input is noiseless, so every error measured downstream is manufactured
    by the pipeline rather than inherited from the data.
    """
    clean = generate_ecg(duration_s=duration_s, fs=FS)
    drive = np.tile(clean, (width, 1)).T
    v_min, v_max = float(np.min(drive)), float(np.max(drive))
    if v_max <= v_min:
        v_max = v_min + 1.0

    net = Network.feedforward(
        layers=[width, width],
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=20.0, tau_minus=20.0),
        seed=seed,
        init_scheme="scaled",
    )
    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
        rng=np.random.default_rng(encoder_seed),
    )
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
    calibrate_gain(net, calib_spikes, dt=DT, target_rate_hz=target_rate_hz, max_iter=22)

    net.reset_transient_state()
    captured: list[list[np.ndarray]] = [[], []]
    for t in range(drive.shape[0]):
        for idx, spikes in enumerate(net.step(encoder.encode(drive[t], DT), DT, False)):
            captured[idx].append(spikes)

    return clean, [
        smooth_traces(np.stack(block, axis=0), DT, tau_decode) for block in captured
    ]


def analyse(
    width: int,
    seed: int,
    duration_s: float,
    tau_decode: float,
    target_rate_hz: float,
) -> list[NoiseRow]:
    """Decompose the error structure of every stage at one width."""
    clean, traces_a = capture(
        width, seed, duration_s, tau_decode, target_rate_hz, encoder_seed=seed
    )
    # Same network, same analog drive, different spike realisation. Error that
    # survives this is a deterministic function of the drive: distortion, not
    # noise, and no amount of averaging removes it.
    _, traces_b = capture(
        width, seed, duration_s, tau_decode, target_rate_hz, encoder_seed=seed + 500_000
    )

    rows: list[NoiseRow] = []
    for order, (name, trace_a, trace_b) in enumerate(
        zip(["encoded", "layer1"], traces_a, traces_b, strict=True)
    ):
        signal, error_a = residuals(trace_a, clean)
        _, error_b = residuals(trace_b, clean)
        var_signal = float(np.var(signal.mean(axis=1)))
        var_single, var_mean, corr = noise_correlation(error_a)

        # Correlation between the two realisations, per neuron, is the share of
        # each neuron's error that is repeatable given the drive.
        sd_a = np.std(error_a, axis=0)
        sd_b = np.std(error_b, axis=0)
        usable = (sd_a > 0.0) & (sd_b > 0.0)
        if np.any(usable):
            cross = np.mean(error_a[:, usable] * error_b[:, usable], axis=0)
            repeatable = float(np.mean(cross / (sd_a[usable] * sd_b[usable])))
        else:
            repeatable = float("nan")

        rows.append(
            NoiseRow(
                width=width,
                stage=name,
                order=order + 1,
                seed=seed,
                n_units=int(trace_a.shape[1]),
                var_signal=var_signal,
                var_single=var_single,
                var_mean=var_mean,
                noise_corr=corr,
                lambda_meas_db=linear_to_db(var_signal / var_mean)
                if var_mean > 0.0
                else math.inf,
                lambda_indep_db=lambda_at_width(var_signal, var_single, 0.0, width),
                lambda_ceiling_db=lambda_at_width(
                    var_signal, var_single, corr, math.inf
                ),
                repeatable_fraction=repeatable,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=TAU_DECODE)
    parser.add_argument("--target-rate", type=float, default=TARGET_RATE_HZ)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512]
    )
    parser.add_argument(
        "--predict-from",
        type=int,
        default=32,
        help="width whose measured c predicts Lambda at every other width",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/m3_bound/m3_noise_structure.csv")
    )
    args = parser.parse_args()

    print(
        f"M3 — error structure   (layers=[N,N], tau={args.tau:.0f}ms, "
        f"rate={args.target_rate:.0f}Hz, clean input, {args.seeds} seeds)"
    )
    print("Lambda is the intrinsic SNR of the stage's population readout.\n")

    rows: list[NoiseRow] = []
    for width in args.widths:
        for seed in range(args.seeds):
            rows.extend(analyse(width, seed, args.duration, args.tau, args.target_rate))

    for stage in ["encoded", "layer1"]:
        print(f"\n### {stage}\n")
        print(
            f"{'N':>5} {'Lambda meas':>12} {'if c=0':>9} {'c':>8} "
            f"{'N->inf':>9} {'repeatable':>11}"
        )
        print("-" * 58)
        for width in args.widths:
            block = [r for r in rows if r.width == width and r.stage == stage]
            print(
                f"{width:>5} {np.mean([r.lambda_meas_db for r in block]):>12.2f} "
                f"{np.mean([r.lambda_indep_db for r in block]):>9.2f} "
                f"{np.mean([r.noise_corr for r in block]):>8.4f} "
                f"{np.mean([r.lambda_ceiling_db for r in block]):>9.2f} "
                f"{np.mean([r.repeatable_fraction for r in block]) * 100:>10.1f}%"
            )

        anchor = [r for r in rows if r.width == args.predict_from and r.stage == stage]
        if not anchor:
            continue
        c_anchor = float(np.mean([r.noise_corr for r in anchor]))
        print(
            f"\n  Predicting every width from c measured at N={args.predict_from} "
            f"(c={c_anchor:.4f}):\n"
        )
        print(f"{'N':>5} {'predicted':>10} {'measured':>9} {'residual':>9}")
        print("-" * 35)
        for width in args.widths:
            block = [r for r in rows if r.width == width and r.stage == stage]
            predicted = float(
                np.mean(
                    [
                        lambda_at_width(r.var_signal, r.var_single, c_anchor, width)
                        for r in block
                    ]
                )
            )
            measured = float(np.mean([r.lambda_meas_db for r in block]))
            print(
                f"{width:>5} {predicted:>10.2f} {measured:>9.2f} "
                f"{measured - predicted:>+9.2f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(NoiseRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
