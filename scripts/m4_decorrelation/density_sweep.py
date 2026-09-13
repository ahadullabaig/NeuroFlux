"""
M4 — is it decorrelation, or is it fan-in?

M3 traced the ceiling to the cosine similarity of weight columns: every
non-negative uniform initialisation gives `cos -> 3/4`, and
`init_weights(scheme="sparse")` at density `rho` lowers it to about `3 rho / 4`,
lifting `[512, 512]` from 3.87 dB to 5.00 dB. But sparsifying does two things at
once. It decorrelates the columns, and it also shrinks every neuron's fan-in to
`K = rho * N`. The two predict different things:

  decorrelation   Lambda is governed by column cosine. At fixed width it rises
                  monotonically as K falls, because cos = 3K/(4N) falls with it.

  fan-in          Each neuron averages K input spike trains, so its own drive
                  gets noisier as K falls. Past some point that noise outweighs
                  the decorrelation, and Lambda turns over at an interior K.

The Bernoulli mask in `init_weights` cannot separate them: fan-in varies from
neuron to neuron, and at `rho = 0.1, N = 32` a neuron sees about three inputs.
This script builds the mask with an **exact** fan-in K per neuron and sweeps K
at each fixed width N, so the two variables move independently. Then:

  - if curves at different N collapse when plotted against cosine, the effect
    is decorrelation;
  - if they collapse against K, it is fan-in;
  - an interior optimum in K locates where one gives way to the other.

Everything else is M3's protocol unchanged: clean input (so the measured SNR is
the layer's intrinsic Lambda), per-layer rate calibration to 40 Hz, error
correlation measured by residualising each neuron's trace against the clean
signal. `K = N` is the dense `uniform` reference at that width.

Usage:
    uv run python scripts/m4_decorrelation/density_sweep.py
    uv run python scripts/m4_decorrelation/density_sweep.py --dataset mit-bih \\
        --records 100 214 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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
from neuroflux_datasets.signals import (
    DEFAULT_CACHE_DIR,
    PHASE3_RECORDS,
    load_clean_ecg,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m3_bound"))
from weight_correlation import (  # noqa: E402
    CALIBRATION_S,
    DT,
    FS,
    MAX_RATE_HZ,
    TARGET_RATE_HZ,
    TAU_DECODE,
    error_structure,
    mean_column_cosine,
    smooth_traces,
)

WIDTHS = [128, 512]
FAN_INS = [4, 8, 16, 32, 64, 128, 256, 512]


@dataclass(frozen=True)
class FanInRow:
    """One (unit, width, fan-in, seed) measurement."""

    unit: str
    width: int
    fan_in: int
    density: float
    seed: int
    cosine_predicted: float
    cosine_measured: float
    error_corr: float
    lambda_db: float
    output_snr_db: float
    mean_rate_hz: float


def fixed_fan_in_weights(
    n_pre: int, n_post: int, fan_in: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Uniform[0, 1] weights where every post-synaptic neuron has exactly `fan_in`
    non-zero inputs, chosen without replacement.

    A Bernoulli mask gives each neuron a binomial fan-in; at small densities the
    spread is large relative to the mean, and a neuron that happens to draw one
    or two inputs contaminates any conclusion about decorrelation. Fixing the
    count removes that variance.
    """
    if not 1 <= fan_in <= n_pre:
        raise ValueError(f"fan_in must be in [1, {n_pre}], got {fan_in}")
    weights = np.zeros((n_pre, n_post), dtype=np.float32)
    for post in range(n_post):
        chosen = rng.choice(n_pre, size=fan_in, replace=False)
        weights[chosen, post] = rng.uniform(0.0, 1.0, size=fan_in)
    return weights


def run_fan_in(
    width: int,
    fan_in: int,
    seed: int,
    duration_s: float,
    dataset: str,
    unit: str,
    cache_dir: str,
) -> FanInRow:
    """Build `[width, width]` with exact fan-in and measure its intrinsic SNR."""
    clean = load_clean_ecg(
        dataset, unit, duration_s=duration_s, fs=FS, window=seed, cache_dir=cache_dir
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
        init_scheme="uniform",
    )
    weights = fixed_fan_in_weights(
        width, width, fan_in, np.random.default_rng(seed + 20_000)
    )
    synapse = net.synapses[0]
    synapse.weights = weights.copy()
    # `reset_state` restores from this copy, so it must carry the new matrix
    # rather than the dense one the constructor drew.
    synapse.initial_weights_copy = weights.copy()

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
    captured = [
        net.step(encoder.encode(drive[t], DT), DT, False)[-1]
        for t in range(drive.shape[0])
    ]
    spikes = np.stack(captured, axis=0)

    traces = smooth_traces(spikes, DT, TAU_DECODE)
    corr, lam = error_structure(traces, clean)
    rate_hz = traces / (TAU_DECODE / 1000.0)
    decoded = (v_min + np.clip(rate_hz / MAX_RATE_HZ, 0.0, 1.0) * (v_max - v_min)).mean(
        axis=1
    )

    density = fan_in / width
    return FanInRow(
        unit=unit,
        width=width,
        fan_in=fan_in,
        density=density,
        seed=seed,
        # Uniform[0,1] on a K-of-N support: mu = rho/2, E[w^2] = rho/3.
        cosine_predicted=0.75 * density,
        cosine_measured=mean_column_cosine(weights.astype(np.float64)),
        error_corr=corr,
        lambda_db=lam,
        output_snr_db=affine_corrected_snr(clean, decoded),
        mean_rate_hz=float(np.mean(spikes) * 1000.0 / DT),
    )


def summarise(rows: list[FanInRow]) -> None:
    """Per-width table, then the collapse test."""
    for width in sorted({r.width for r in rows}):
        print(f"\n[{width}, {width}]")
        print(
            f"  {'K':>4} {'rho':>6} {'cos pred':>8} {'cos meas':>8} "
            f"{'err corr':>8} {'Lambda dB':>14} {'out SNR dB':>14} {'rate':>6}"
        )
        for fan_in in sorted({r.fan_in for r in rows if r.width == width}):
            group = [r for r in rows if r.width == width and r.fan_in == fan_in]
            unit_means = _unit_means(group, "lambda_db")
            snr_means = _unit_means(group, "output_snr_db")
            print(
                f"  {fan_in:>4} {fan_in / width:>6.3f} "
                f"{np.mean([r.cosine_predicted for r in group]):>8.4f} "
                f"{np.mean([r.cosine_measured for r in group]):>8.4f} "
                f"{np.nanmean([r.error_corr for r in group]):>8.3f} "
                f"{np.mean(unit_means):>7.2f} ±{_sd(unit_means):>5.2f} "
                f"{np.mean(snr_means):>7.2f} ±{_sd(snr_means):>5.2f} "
                f"{np.mean([r.mean_rate_hz for r in group]):>6.1f}"
            )
        best = max(
            sorted({r.fan_in for r in rows if r.width == width}),
            key=lambda k: float(
                np.mean(
                    _unit_means(
                        [r for r in rows if r.width == width and r.fan_in == k],
                        "output_snr_db",
                    )
                )
            ),
        )
        print(f"  best fan-in at this width: K = {best} (rho = {best / width:.3f})")


def _unit_means(rows: list[FanInRow], metric: str) -> list[float]:
    """
    Average seeds within each unit, so the spread is across records.

    With a single unit (synthetic) there is nothing to average across, so the
    per-seed values are returned instead and the spread is seed-to-seed.
    """
    by_unit: dict[str, list[float]] = {}
    for row in rows:
        by_unit.setdefault(row.unit, []).append(float(getattr(row, metric)))
    if len(by_unit) == 1:
        return next(iter(by_unit.values()))
    return [float(np.mean(values)) for values in by_unit.values()]


def _sd(values: list[float]) -> float:
    """Sample standard deviation, or 0 for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    """Sweep exact fan-in at each width and print the dissociation tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--fan-ins", type=int, nargs="+", default=FAN_INS)
    parser.add_argument(
        "--dataset", choices=["synthetic", "mit-bih"], default="synthetic"
    )
    parser.add_argument("--records", nargs="*", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("results/m4_decorrelation/density_sweep.csv")
    )
    args = parser.parse_args()

    units = list(args.records) if args.dataset == "mit-bih" else ["synthetic"]
    rows: list[FanInRow] = []
    for width in args.widths:
        for fan_in in [k for k in args.fan_ins if k <= width]:
            for unit in units:
                for seed in range(args.seeds):
                    rows.append(
                        run_fan_in(
                            width,
                            fan_in,
                            seed,
                            args.duration,
                            args.dataset,
                            unit,
                            args.cache_dir,
                        )
                    )
            print(f"  width {width} fan-in {fan_in} done", flush=True)

    summarise(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(FanInRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {args.out}")
    if any(math.isnan(r.error_corr) for r in rows):
        print("Note: some rows have undefined error correlation (single unit).")


if __name__ == "__main__":
    main()
