"""
M3 — the analytical source of the shared noise: non-negative weights.

`noise_structure.py` measures a mean pairwise error correlation of c ~ 0.33-0.47
between the neurons of a LIF layer, against c ~ 0.07 at the encoder, and shows
that c alone caps what any width can buy. This script derives where that number
comes from, and it turns out to be fixed by the weight initialisation before a
single spike is simulated.

Neuron `i`'s drive is `d_i = sum_k W[k,i] * s_k`. For a spike vector whose
components are roughly independent, the correlation between two neurons' drives
is the **cosine similarity of their weight columns**:

    corr(d_i, d_j) = (w_i . w_j) / (|w_i| |w_j|)

and for iid weight entries with mean `mu` and variance `sigma^2` that is

    corr -> mu^2 / (mu^2 + sigma^2).

Every initialisation scheme in `network/topology.py` draws from a **non-negative**
range, because STDP bounds weights to `[w_min, w_max] = [0, 1]`. For any uniform
draw on `[0, a]`, `mu = a/2` and `sigma^2 = a^2/12`, so

    corr -> (a/2)^2 / ((a/2)^2 + a^2/12) = 3/4

**independently of `a`.** The scale cancels. Three consequences:

  1. Gain calibration cannot touch this. Calibration rescales weights; the
     correlation is scale-invariant. That is why M1b's per-layer calibration
     fixed the degeneracy without moving the ceiling.
  2. Every neuron in a layer computes nearly the same positively-weighted
     average of its inputs, so a layer of N neurons has far fewer than N
     effective degrees of freedom, and the population average converges to
     something noisier than `1/N` scaling promises.
  3. The fix is to decorrelate the columns, which means lowering `mu^2/sigma^2`.
     `init_weights(scheme="sparse")` already does exactly that — at density
     `rho` the ratio falls by roughly `rho` — and it is already in the library,
     tested and unused.

This script checks the 3/4 prediction against the sampled matrices, then runs the
full pipeline under each scheme to see what the predicted decorrelation is
actually worth in dB.

Usage:
    uv run python scripts/m3_bound/weight_correlation.py
    uv run python scripts/m3_bound/weight_correlation.py --width 256 --seeds 5
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
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_core.network.topology import InitScheme, init_weights
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, load_clean_ecg

FS = 250.0
DT = 1000.0 / FS
TAU_DECODE = 20.0
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0

SCHEMES: list[InitScheme] = ["uniform", "scaled", "gaussian", "sparse"]


@dataclass(frozen=True)
class SchemeRow:
    """One init scheme at one seed: predicted correlation and what it bought."""

    scheme: str
    width: int
    unit: str
    seed: int
    weight_mean: float
    weight_std: float
    cosine_predicted: float
    cosine_measured: float
    error_corr: float
    lambda_db: float
    output_snr_db: float
    mean_rate_hz: float


def mean_column_cosine(weights: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity between columns, without the N x N matrix.

    The Gram matrix of unit-normalised columns sums to `|sum_i u_i|^2`, and its
    diagonal is N, so the off-diagonal mean follows directly.
    """
    norms = np.linalg.norm(weights, axis=0)
    usable = norms > 0.0
    unit = weights[:, usable] / norms[usable]
    n = int(unit.shape[1])
    if n < 2:
        return float("nan")
    total = float(np.sum(np.square(np.sum(unit, axis=1))))
    return (total - n) / (n * (n - 1))


def smooth_traces(spikes: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Per-neuron exponential spike traces, matching PopulationDecoder."""
    decay = math.exp(-dt / tau)
    traces = np.empty(spikes.shape, dtype=np.float64)
    state = np.zeros(spikes.shape[1], dtype=np.float64)
    for t in range(spikes.shape[0]):
        state = state * decay + spikes[t].astype(np.float64)
        traces[t] = state
    return traces


def error_structure(traces: np.ndarray, clean: np.ndarray) -> tuple[float, float]:
    """
    Mean pairwise error correlation and the resulting intrinsic SNR, in dB.

    Mirrors `noise_structure.py`: strip each neuron's linearly-predictable
    signal component, then compare the population-mean error variance against
    the per-neuron variance.
    """
    design = np.vstack([clean, np.ones_like(clean)]).T
    coeffs, *_ = np.linalg.lstsq(design, traces, rcond=None)
    signal = design @ coeffs
    error = traces - signal

    n_units = traces.shape[1]
    var_single = float(np.mean(np.var(error, axis=0)))
    var_mean = float(np.var(error.mean(axis=1)))
    var_signal = float(np.var(signal.mean(axis=1)))

    if n_units < 2 or var_single <= 0.0:
        return float("nan"), math.inf
    corr = (n_units * var_mean / var_single - 1.0) / (n_units - 1)
    lam = linear_to_db(var_signal / var_mean) if var_mean > 0.0 else math.inf
    return float(corr), lam


def run_scheme(
    scheme: InitScheme,
    width: int,
    seed: int,
    duration_s: float,
    tau_decode: float,
    dataset: str = "synthetic",
    unit: str = "synthetic",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> SchemeRow:
    """Build `[width, width]` under one init scheme and measure what it delivers."""
    clean = load_clean_ecg(
        dataset,
        unit,
        duration_s=duration_s,
        fs=FS,
        window=seed,
        cache_dir=cache_dir,
    )
    drive = np.tile(clean, (width, 1)).T
    v_min, v_max = float(np.min(drive)), float(np.max(drive))
    if v_max <= v_min:
        v_max = v_min + 1.0

    net = Network.feedforward(
        layers=[width, width],
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=20.0, tau_minus=20.0),
        seed=seed,
        init_scheme=scheme,
    )
    weights = np.asarray(net.synapses[0].weights, dtype=np.float64)

    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
        rng=np.random.default_rng(seed),
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
    calibrate_gain(net, calib_spikes, dt=DT, target_rate_hz=TARGET_RATE_HZ, max_iter=22)

    net.reset_transient_state()
    captured: list[np.ndarray] = []
    for t in range(drive.shape[0]):
        captured.append(net.step(encoder.encode(drive[t], DT), DT, False)[-1])
    spikes = np.stack(captured, axis=0)

    traces = smooth_traces(spikes, DT, tau_decode)
    corr, lam = error_structure(traces, clean)
    rate_hz = traces / (tau_decode / 1000.0)
    decoded = (v_min + np.clip(rate_hz / MAX_RATE_HZ, 0.0, 1.0) * (v_max - v_min)).mean(
        axis=1
    )

    w_mean = float(np.mean(weights))
    w_std = float(np.std(weights))
    return SchemeRow(
        scheme=str(scheme),
        width=width,
        unit=unit,
        seed=seed,
        weight_mean=w_mean,
        weight_std=w_std,
        cosine_predicted=w_mean**2 / (w_mean**2 + w_std**2),
        cosine_measured=mean_column_cosine(weights),
        error_corr=corr,
        lambda_db=lam,
        output_snr_db=affine_corrected_snr(clean, decoded),
        mean_rate_hz=float(np.mean(spikes) * 1000.0 / DT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=TAU_DECODE)
    parser.add_argument(
        "--dataset", choices=["synthetic", "mit-bih"], default="synthetic"
    )
    parser.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=None,
        help="MIT-BIH records to average over (default: 214)",
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("results/m3_bound/m3_weight_correlation.csv")
    )
    args = parser.parse_args()

    print("M3 — weight-column cosine similarity sets the shared-noise floor")
    print(
        f"(layers=[{args.width},{args.width}], tau={args.tau:.0f}ms, clean input, "
        f"{args.dataset}, {args.seeds} seeds)\n"
    )

    # Step 1: pure matrix algebra, no simulation. The 3/4 prediction.
    print("Step 1 — weight statistics alone, no simulation:\n")
    print(f"{'scheme':>10} {'mean':>8} {'std':>8} {'cos pred':>9} {'cos meas':>9}")
    print("-" * 48)
    for scheme in SCHEMES:
        w = init_weights(
            shape=(args.width, args.width),
            scheme=scheme,
            rng=np.random.default_rng(0),
            w_min=0.0,
            w_max=1.0,
        ).astype(np.float64)
        mu, sd = float(np.mean(w)), float(np.std(w))
        print(
            f"{scheme:>10} {mu:>8.4f} {sd:>8.4f} "
            f"{mu**2 / (mu**2 + sd**2):>9.4f} {mean_column_cosine(w):>9.4f}"
        )

    # Step 2: what that correlation is worth once the network actually runs.
    print("\nStep 2 — the same schemes through the full pipeline:\n")
    print(
        f"{'scheme':>10} {'cos':>7} {'err corr':>9} {'Lambda dB':>10} "
        f"{'SNR out':>9} {'rate Hz':>8}"
    )
    print("-" * 57)
    units = (
        (list(args.records) if args.records else ["214"])
        if args.dataset == "mit-bih"
        else ["synthetic"]
    )
    rows: list[SchemeRow] = []
    for scheme in SCHEMES:
        block = [
            run_scheme(
                scheme,
                args.width,
                seed,
                args.duration,
                args.tau,
                dataset=args.dataset,
                unit=unit,
                cache_dir=args.cache_dir,
            )
            for unit in units
            for seed in range(args.seeds)
        ]
        rows.extend(block)
        print(
            f"{scheme:>10} {np.mean([r.cosine_measured for r in block]):>7.3f} "
            f"{np.mean([r.error_corr for r in block]):>9.4f} "
            f"{np.mean([r.lambda_db for r in block]):>10.2f} "
            f"{np.mean([r.output_snr_db for r in block]):>9.2f} "
            f"{np.mean([r.mean_rate_hz for r in block]):>8.1f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(SchemeRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
