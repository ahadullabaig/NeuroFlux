"""Passive (untrained) R-peak preservation regression.

Pins one of the load-bearing Phase 3 diagnostic findings — that the
*passive* SNN pipeline (no STDP learning) preserves R-peak landmarks
with very high fidelity by virtue of:

- Poisson rate encoding faithfully passing the temporal envelope through
- LIF membrane integration acting as a low-pass smoother
- PopulationDecoder smoothing the output spike rate back to analog

If this property regresses, the original Phase 3 conclusion — that
trained NeuroFlux's value-add over the passive baseline is *amplitude
domain* (ΔSNR) rather than *timing domain* (rpeak) — no longer holds,
and the Phase 4 baseline comparison framework would need rethinking.

The test uses a small synthetic ECG (no MIT-BIH dependency) so it runs
in CI without bundled data. It exercises ``run_single_experiment`` with
``epochs=0`` — the same code path used by ``configs/phase3/
diagnostic_untrained.yaml`` — and computes ``rpeak_integrity`` directly
on the synthetic clean / reconstructed pair.

See ``docs/phase3_findings.md`` §5.6.2 for the original measurement
(rpeak = 1.000 across all 16 MIT-BIH records and all input SNRs).
"""

from __future__ import annotations

import numpy as np
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.config import LIFConfig, STDPConfig
from neuroflux_core.metrics.rpeak import rpeak_integrity
from neuroflux_core.network.feedforward import FeedForwardNetwork, Network
from neuroflux_core.network.simulator import Simulator
from neuroflux_datasets.synthetic import SignalGenerator


def _build_passive_pipeline(
    clean_min: float, clean_max: float, fs: float, seed: int
) -> Simulator:
    """Construct the canonical 32→16→8 SNN with no STDP training."""
    net = Network.feedforward(
        layers=[32, 16, 8],
        neuron=LIFConfig(),
        plasticity=STDPConfig(),
        seed=seed,
    )
    encoder = PoissonRateEncoder(
        v_min=clean_min,
        v_max=clean_max,
        max_firing_rate_hz=100.0,
        rng=np.random.default_rng(seed),
    )
    decoder = PopulationDecoder(
        v_min=clean_min, v_max=clean_max, max_firing_rate_hz=100.0
    )
    return Simulator(net, dt=1000.0 / fs, seed=seed, encoder=encoder, decoder=decoder)


def test_passive_pipeline_preserves_rpeaks_clean_ecg() -> None:
    """Untrained pipeline on a clean synthetic ECG must hit rpeak ≥ 0.95.

    Phase 3 diagnostic measured rpeak = 1.000 across all real MIT-BIH
    records — synthetic ECG should be at least as easy. The 0.95 bar
    leaves headroom for ECG morphology variations and seed jitter while
    still catching any change to LIF / encoder / decoder that breaks the
    envelope-preservation property.
    """
    fs = 250.0
    duration_s = 4.0
    seed = 0

    gen = SignalGenerator(fs=fs, seed=seed)
    clean, _noisy = gen.generate(
        signal_type="ecg", duration_s=duration_s, flicker_amplitude=0.0
    )

    v_min, v_max = float(np.min(clean)), float(np.max(clean))
    if v_max <= v_min:
        v_max = v_min + 1.0

    sim = _build_passive_pipeline(v_min, v_max, fs, seed)
    assert isinstance(sim.network, FeedForwardNetwork)

    # Tile single-channel input across 32 input neurons — same as
    # ``run_single_experiment``.
    tiled = np.tile(clean, (32, 1)).T
    out_spikes = sim.run(tiled, duration_ms=duration_s * 1000.0, training=False)
    reconstructed = sim.decode(out_spikes)

    rpeak = rpeak_integrity(clean, reconstructed, fs=fs)
    assert rpeak >= 0.95, (
        f"Passive pipeline lost R-peaks: rpeak_integrity={rpeak:.3f} "
        "(expected ≥ 0.95). This breaks the Phase 3 finding that the "
        "untrained SNN preserves clinical landmarks — review changes to "
        "LIF dynamics, Poisson encoding, or population decoding."
    )


def test_passive_pipeline_preserves_rpeaks_noisy_ecg() -> None:
    """Same passive pipeline must preserve R-peaks under 50 Hz + 10 dB SNR.

    The Phase 3 diagnostic showed rpeak = 1.000 even at 0 dB input SNR.
    Testing at a more forgiving 10 dB lets the test be insensitive to
    seed-dependent Poisson sampling while still asserting the property
    survives realistic interference.
    """
    fs = 250.0
    duration_s = 4.0
    seed = 0

    gen = SignalGenerator(fs=fs, seed=seed)
    clean, noisy = gen.generate(
        signal_type="ecg",
        duration_s=duration_s,
        flicker_freq=50.0,
        flicker_amplitude=0.5,
        snr_db=10.0,
    )

    v_min, v_max = float(np.min(noisy)), float(np.max(noisy))
    if v_max <= v_min:
        v_max = v_min + 1.0

    sim = _build_passive_pipeline(v_min, v_max, fs, seed)
    assert isinstance(sim.network, FeedForwardNetwork)

    tiled = np.tile(noisy, (32, 1)).T
    out_spikes = sim.run(tiled, duration_ms=duration_s * 1000.0, training=False)
    reconstructed = sim.decode(out_spikes)

    rpeak = rpeak_integrity(clean, reconstructed, fs=fs)
    assert rpeak >= 0.95, (
        f"Passive pipeline lost R-peaks under noise: "
        f"rpeak_integrity={rpeak:.3f} (expected ≥ 0.95). "
        "See docs/phase3_findings.md §5.6.2 — untrained rpeak should be "
        "perfect or near-perfect across all input SNRs."
    )
