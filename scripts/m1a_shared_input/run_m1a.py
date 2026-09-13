"""
M1a — the shared-input control.

R0 found that the Finding 4 capacity sweep cannot test what it claimed to.
`cli.py` tiles a single noisy scalar across every input channel
(`np.tile(noisy, (n_input, 1)).T`), so the information the input carries about
the clean signal is one sample's worth no matter how wide the input layer is.
Moreno-Bote et al. (2014, Nat. Neurosci. 17:1410) prove Fisher information
*must* saturate under exactly that condition: "if neurons share input, the
resulting shared variability in their response cannot be averaged out."

So the measured flatness may be the predicted consequence of the input
construction rather than evidence of an architectural ceiling. This script
distinguishes the two by re-running the width sweep under three input modes:

  tiled        Current behaviour. Every channel carries the same noisy scalar.
               Control. Theory predicts a flat ceiling here.

  independent  Same clean signal, `n_input` independent noise realisations.
               The clean information-theoretic control: information genuinely
               grows with width, and an ideal estimator gains 10*log10(N) dB,
               i.e. +3.01 dB per doubling. This is the mode that tests the
               hypothesis.

  delay        Delay-line embedding — channel k carries the noisy signal
               lagged by k samples. The realistic engineering variant: it adds
               temporal context rather than repeated measurements of the
               current sample, so the +3 dB/doubling reference does not apply
               to it directly.

Every condition is run under per-layer rate calibration (`calibrate_gain`), so
width is not confounded with operating point, and every reconstruction is
gated on the degeneracy guard before being scored.

Interpretation is pre-registered in docs/paper_roadmap.md R4/M1a. Do not
change it after seeing these numbers.

On MIT-BIH the unit of analysis is the record, per `docs/r2_findings.md`: each
seed takes a different artifact-free window of the record, and the summary
slope is computed per record before it is averaged.

Usage:
    uv run python scripts/m1a_shared_input/run_m1a.py
    uv run python scripts/m1a_shared_input/run_m1a.py --seeds 5 --quick
    uv run python scripts/m1a_shared_input/run_m1a.py --dataset mit-bih --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from neuroflux_core import (
    LIFConfig,
    Network,
    PoissonRateEncoder,
    PopulationDecoder,
    Simulator,
    STDPConfig,
    calibrate_gain,
    measure_layer_rates,
)
from neuroflux_core.metrics.degeneracy import is_degenerate, output_std
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_datasets.signals import (
    DEFAULT_CACHE_DIR,
    PHASE3_RECORDS,
    load_clean_ecg,
)
from neuroflux_datasets.synthetic import add_gaussian_noise

InputMode = Literal["tiled", "independent", "delay"]
INPUT_MODES: tuple[InputMode, ...] = get_args(InputMode)

FS = 250.0
DT = 1000.0 / FS
TARGET_RATE_HZ = 40.0
CALIBRATION_S = 2.0


@dataclass(frozen=True)
class Result:
    """One (unit, mode, width, seed) measurement."""

    unit: str
    mode: str
    n_input: int
    seed: int
    input_snr_db: float
    ideal_snr_db: float
    ceiling_db: float
    raw_snr_db: float
    out_std: float
    degenerate: bool
    gain_layer0: float
    gain_layer1: float
    rate_layer0: float
    rate_layer1: float


def _lag(signal: np.ndarray, k: int) -> np.ndarray:
    """Shift `signal` forward by k samples, holding the first value at the edge."""
    if k == 0:
        return signal.copy()
    out = np.empty_like(signal)
    out[:k] = signal[0]
    out[k:] = signal[:-k]
    return out


def build_input(
    mode: InputMode,
    clean: np.ndarray,
    noisy: np.ndarray,
    n_input: int,
    input_snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct the (T, n_input) drive for one input mode."""
    if mode == "tiled":
        return np.tile(noisy, (n_input, 1)).T
    if mode == "independent":
        return np.stack(
            [add_gaussian_noise(clean, input_snr_db, rng) for _ in range(n_input)],
            axis=1,
        )
    if mode == "delay":
        return np.stack([_lag(noisy, k) for k in range(n_input)], axis=1)
    raise ValueError(f"Unknown input mode: {mode!r}")


def run_condition(
    mode: InputMode,
    n_input: int,
    seed: int,
    input_snr_db: float,
    duration_s: float,
    epochs: int,
    hidden: list[int],
    dataset: str = "synthetic",
    unit: str = "synthetic",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Result:
    """Calibrate, run and score a single condition."""
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

    drive = build_input(mode, clean, noisy, n_input, input_snr_db, rng)

    v_min = float(np.min(drive))
    v_max = float(np.max(drive))
    if v_max <= v_min:
        v_max = v_min + 1.0

    layers = [n_input, *hidden]
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
        max_firing_rate_hz=100.0,
        rng=np.random.default_rng(seed),
    )
    decoder = PopulationDecoder(v_min=v_min, v_max=v_max, max_firing_rate_hz=100.0)

    # Calibrate on a short representative window, using spikes from the same
    # encoder settings the run will use. A separate RNG keeps calibration from
    # consuming the encoder's stream.
    calib_steps = int(CALIBRATION_S * FS)
    calib_rng = np.random.default_rng(seed + 10_000)
    calib_encoder = PoissonRateEncoder(
        v_min=v_min, v_max=v_max, max_firing_rate_hz=100.0, rng=calib_rng
    )
    calib_spikes = np.stack(
        [calib_encoder.encode(drive[t], DT) for t in range(calib_steps)], axis=0
    )
    gains = calibrate_gain(
        net, calib_spikes, dt=DT, target_rate_hz=TARGET_RATE_HZ, max_iter=22
    )
    rates = measure_layer_rates(net, calib_spikes, dt=DT)

    sim = Simulator(net, dt=DT, seed=seed, encoder=encoder, decoder=decoder)
    duration_ms = duration_s * 1000.0

    for _ in range(epochs):
        sim.run(drive, duration_ms=duration_ms, training=True)
        net.reset_transient_state()

    out_spikes = sim.run(drive, duration_ms=duration_ms, training=False)
    reconstructed = sim.decode(out_spikes)

    degenerate = is_degenerate(reconstructed)
    ceiling = math.nan if degenerate else affine_corrected_snr(clean, reconstructed)
    raw = math.nan if degenerate else calculate_snr(clean, reconstructed)

    # An ideal estimator averaging N independent observations gains 10log10(N).
    ideal = (
        input_snr_db + 10.0 * math.log10(n_input)
        if mode == "independent"
        else input_snr_db
    )

    return Result(
        unit=unit,
        mode=mode,
        n_input=n_input,
        seed=seed,
        input_snr_db=input_snr_db,
        ideal_snr_db=ideal,
        ceiling_db=ceiling,
        raw_snr_db=raw,
        out_std=output_std(reconstructed),
        degenerate=degenerate,
        gain_layer0=gains[0],
        gain_layer1=gains[1] if len(gains) > 1 else math.nan,
        rate_layer0=rates[0],
        rate_layer1=rates[1] if len(rates) > 1 else math.nan,
    )


def slope_per_doubling(widths: list[int], values: list[float]) -> float:
    """Least-squares dB gained per doubling of width, ignoring NaNs."""
    pairs = [(w, v) for w, v in zip(widths, values, strict=True) if not math.isnan(v)]
    if len(pairs) < 2:
        return math.nan
    log_w = np.log2(np.array([p[0] for p in pairs], dtype=np.float64))
    vals = np.array([p[1] for p in pairs], dtype=np.float64)
    return float(np.polyfit(log_w, vals, 1)[0])


def per_unit_slopes(results: list[Result], widths: list[int]) -> dict[str, float]:
    """
    Slope per doubling computed separately for each record.

    Pooling every record into one mean per width and fitting once would let a
    few high-ceiling records set the slope for all of them. Fitting per record
    and reporting the spread keeps the record as the unit of analysis.
    """
    slopes: dict[str, float] = {}
    for unit in sorted({r.unit for r in results}):
        means: list[float] = []
        for width in widths:
            good = [
                r.ceiling_db
                for r in results
                if r.unit == unit and r.n_input == width and not r.degenerate
            ]
            means.append(float(np.mean(good)) if good else math.nan)
        slopes[unit] = slope_per_doubling(widths, means)
    return slopes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--input-snr", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="widths 8-64 only")
    parser.add_argument(
        "--dataset", choices=["synthetic", "mit-bih"], default="synthetic"
    )
    parser.add_argument("--records", nargs="*", default=list(PHASE3_RECORDS))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/m1a_shared_input/m1a_results.csv"),
    )
    args = parser.parse_args()

    widths = [8, 16, 32, 64] if args.quick else [8, 16, 32, 64, 128, 256]
    hidden = [16, 8]
    units = list(args.records) if args.dataset == "mit-bih" else ["synthetic"]

    print(
        f"M1a — shared-input control   (epochs={args.epochs}, "
        f"input SNR={args.input_snr} dB, {args.seeds} seeds, "
        f"duration={args.duration}s, dataset={args.dataset}, "
        f"{len(units)} unit(s))"
    )
    print(
        "Per-layer rate calibration to "
        f"{TARGET_RATE_HZ:.0f} Hz; scores are affine-corrected SNR.\n"
    )

    results: list[Result] = []
    for mode in INPUT_MODES:
        print(f"--- mode: {mode} ---")
        print(
            f"{'N_in':>5} | {'g_L0':>9} {'g_L1':>9} | {'r_L0':>6} {'r_L1':>6} "
            f"| {'ceiling dB':>16} | {'out std':>8} | {'ideal dB':>8}"
        )
        means: list[float] = []
        for n_input in widths:
            per_seed = [
                run_condition(
                    mode,
                    n_input,
                    seed,
                    args.input_snr,
                    args.duration,
                    args.epochs,
                    hidden,
                    dataset=args.dataset,
                    unit=unit,
                    cache_dir=args.cache_dir,
                )
                for unit in units
                for seed in range(args.seeds)
            ]
            results.extend(per_seed)

            good = [r.ceiling_db for r in per_seed if not r.degenerate]
            mean = float(np.mean(good)) if good else math.nan
            std = float(np.std(good)) if good else math.nan
            means.append(mean)
            ref = per_seed[0]
            flag = "  DEGENERATE" if len(good) < len(per_seed) else ""
            print(
                f"{n_input:>5} | {ref.gain_layer0:>9.3f} {ref.gain_layer1:>9.3f} "
                f"| {ref.rate_layer0:>6.1f} {ref.rate_layer1:>6.1f} "
                f"| {mean:>9.2f} ±{std:>5.2f} "
                f"| {float(np.mean([r.out_std for r in per_seed])):>8.4f} "
                f"| {ref.ideal_snr_db:>8.2f}{flag}"
            )

        slope = slope_per_doubling(widths, means)
        finite = [m for m in means if not math.isnan(m)]
        total = finite[-1] - finite[0] if len(finite) >= 2 else math.nan
        reference = " (ideal +3.01)" if mode == "independent" else ""
        print(f"  slope: {slope:+.3f} dB per doubling{reference}")
        print(f"  total {widths[0]} -> {widths[-1]}: {total:+.2f} dB")
        if len(units) > 1:
            unit_slopes = per_unit_slopes(
                [r for r in results if r.mode == mode], widths
            )
            finite_slopes = [v for v in unit_slopes.values() if not math.isnan(v)]
            print(
                f"  per-record slope: {np.mean(finite_slopes):+.3f} "
                f"± {np.std(finite_slopes, ddof=1):.3f} dB per doubling "
                f"[{min(finite_slopes):+.3f}, {max(finite_slopes):+.3f}], "
                f"n={len(finite_slopes)} records"
            )
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"Wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
