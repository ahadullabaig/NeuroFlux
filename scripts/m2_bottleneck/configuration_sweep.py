"""
M2 companion — how the ceiling responds to configuration.

`run_m2.py` decomposes a single configuration stage by stage. This script asks
the complementary question: across configurations, what actually moves the
final number? It runs three sweeps.

  encoder     The Poisson encoder in isolation — no network, no decoder — as a
              function of input SNR, population size and readout time constant.
              Establishes what the encoding step alone can preserve.

  pipeline    The full network across depth, width and tau_decode. Establishes
              what survives, and which knob matters.

  best        The strongest configuration found, against the shipped default.

The headline from these sweeps is that the loss *redistributes* between the
encoder and the LIF layers as tau and width change, while the total stays far
more stable than any single stage. That is why every architectural knob swept
in Phase 3 and Findings 1-5 looked invariant.

Usage:
    uv run python scripts/m2_bottleneck/configuration_sweep.py
    uv run python scripts/m2_bottleneck/configuration_sweep.py --sweep encoder
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from neuroflux_core import (
    LIFConfig,
    Network,
    PoissonRateEncoder,
    PopulationDecoder,
    Simulator,
    STDPConfig,
    calibrate_gain,
)
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from neuroflux_datasets.synthetic import add_gaussian_noise, generate_ecg

FS = 250.0
DT = 1000.0 / FS
MAX_RATE_HZ = 100.0
TARGET_RATE_HZ = 40.0
DURATION_S = 8.0


def _signal(seed: int, input_snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    clean = generate_ecg(duration_s=DURATION_S, fs=FS)
    noisy = add_gaussian_noise(clean, input_snr_db, np.random.default_rng(seed))
    return clean, noisy


def encoder_only(
    n_in: int, tau: float, seed: int, input_snr_db: float, max_rate: float
) -> float:
    """Encode, read back the population rate, score. No network in between."""
    clean, noisy = _signal(seed, input_snr_db)
    drive = np.tile(noisy, (n_in, 1)).T
    encoder = PoissonRateEncoder(
        v_min=float(drive.min()),
        v_max=float(drive.max()),
        max_firing_rate_hz=max_rate,
        rng=np.random.default_rng(seed),
    )
    spikes = np.stack([encoder.encode(drive[t], DT) for t in range(drive.shape[0])])

    decay = math.exp(-DT / tau)
    state = np.zeros(n_in)
    out = np.empty(spikes.shape[0])
    for t in range(spikes.shape[0]):
        state = state * decay + spikes[t]
        out[t] = state.mean()
    return affine_corrected_snr(clean, out)


def full_pipeline(
    layers: list[int], tau: float, seed: int, input_snr_db: float
) -> float:
    """Calibrated, passive, end-to-end. NaN if the output is degenerate."""
    clean, noisy = _signal(seed, input_snr_db)
    drive = np.tile(noisy, (layers[0], 1)).T
    v_min, v_max = float(drive.min()), float(drive.max())

    net = Network.feedforward(
        layers=layers,
        neuron=LIFConfig(),
        plasticity=STDPConfig(),
        seed=seed,
        init_scheme="scaled",
    )
    calib_encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=MAX_RATE_HZ,
        rng=np.random.default_rng(seed + 10_000),
    )
    calib_spikes = np.stack(
        [calib_encoder.encode(drive[t], DT) for t in range(int(2.0 * FS))]
    )
    calibrate_gain(net, calib_spikes, dt=DT, target_rate_hz=TARGET_RATE_HZ, max_iter=22)

    sim = Simulator(
        net,
        dt=DT,
        seed=seed,
        encoder=PoissonRateEncoder(
            v_min=v_min,
            v_max=v_max,
            max_firing_rate_hz=MAX_RATE_HZ,
            rng=np.random.default_rng(seed),
        ),
        decoder=PopulationDecoder(
            v_min=v_min,
            v_max=v_max,
            max_firing_rate_hz=MAX_RATE_HZ,
            tau_decode=tau,
        ),
    )
    recon = sim.decode(sim.run(drive, duration_ms=DURATION_S * 1000.0, training=False))
    return math.nan if is_degenerate(recon) else affine_corrected_snr(clean, recon)


def _mean(values: list[float]) -> float:
    good = [v for v in values if not math.isnan(v)]
    return float(np.mean(good)) if good else math.nan


def sweep_encoder(seeds: int) -> None:
    print("\n=== Encoder in isolation (affine-corrected SNR, dB) ===")
    print("\n  output vs INPUT SNR   (N=32, tau=20 ms, 100 Hz)")
    for snr in (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0):
        actual = _mean(
            [affine_corrected_snr(*_signal(s, snr)[::-1][::-1]) for s in range(seeds)]
        )
        out = _mean([encoder_only(32, 20.0, s, snr, MAX_RATE_HZ) for s in range(seeds)])
        print(f"    nominal {snr:>5.0f} dB (actual {actual:>6.2f}) -> {out:>6.2f}")

    print("\n  output vs N_in x tau   (input 20 dB, 100 Hz)")
    taus = (4.0, 8.0, 20.0)
    print("      N_in | " + " | ".join(f"tau={t:>2.0f}ms" for t in taus))
    for n_in in (32, 128, 512, 4096):
        cells = []
        for tau in taus:
            value = _mean(
                [encoder_only(n_in, tau, s, 20.0, MAX_RATE_HZ) for s in range(seeds)]
            )
            cells.append(f"{value:>7.2f}")
        print(f"    {n_in:>6} | " + " | ".join(cells))


def sweep_pipeline(seeds: int) -> None:
    print("\n=== Full pipeline (affine-corrected SNR, dB; input 20 dB) ===")
    taus = (4.0, 8.0, 12.0, 20.0, 40.0)
    configs = [[128, 128], [512, 512], [128, 128, 128], [32, 16, 8]]
    print("              layers | " + " | ".join(f"tau={t:>2.0f}" for t in taus))
    print("  " + "-" * 66)
    for layers in configs:
        cells = []
        for tau in taus:
            value = _mean([full_pipeline(layers, tau, s, 20.0) for s in range(seeds)])
            cells.append("DEGEN" if math.isnan(value) else f"{value:>5.2f}")
        marker = "   <- shipped default" if layers == [32, 16, 8] else ""
        print(f"  {str(layers):>18} | " + " | ".join(f"{c:>5}" for c in cells) + marker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--sweep", choices=("encoder", "pipeline", "all"), default="all"
    )
    args = parser.parse_args()

    if args.sweep in ("encoder", "all"):
        sweep_encoder(args.seeds)
    if args.sweep in ("pipeline", "all"):
        sweep_pipeline(args.seeds)


if __name__ == "__main__":
    main()
