"""
R3 — one pipeline for every characterisation experiment.

C1 (flatness), C2 (invariances) and C3 (operating region) ask different
questions of the same measurement: run a `[N_in, ..., N_out]` network on a
signal, read the output population out, and score it against the clean
signal. Keeping that measurement in one place is what lets their numbers be
compared with each other and with M3's, so the experiments differ only in the
grid of `Spec`s they build.

**Two operating points.**

  shipped      What `cli.py` runs and Phase 3 reported: uniform weight init,
               one global `g_syn = 30` for every layer, the encoder's ceiling
               fixed at 100 Hz. Every rate follows from the signal and the
               weights, so it drifts with width, with record and with training.

  controlled   What M3 derived the ceiling on: scaled init, the encoder held
               at its target mean rate by `calibrate_encoder_rate`, and every
               LIF layer held at its target by `calibrate_gain`. A trained
               network is **recalibrated after training**, so a learning knob
               changes the weights and nothing else.

**Interference.** Input SNR is set with `synthetic.add_interference`, which
scales the realised interference to the exact target variance against the
clean signal. Gaussian noise is the default; flicker is an interference
*type* compared at matched SNR, not a fixed-amplitude addition that caps the
axis (the C1 confound in `docs/paper_roadmap.md`).

**Pairing.** Within a seed every condition sees the same clean window, the
same interference shape (scaled to each SNR), the same initial weights and
the same encoder stream, and a trained network is evaluated on the same spike
stream as an untrained one. Independent streams for noise, encoder,
calibration and signal come from one `SeedSequence`, so none of them share
bits.

**Scoring.** Output SNR is reported raw and affine-corrected, every row
carries a degeneracy verdict, and every stage (encoder and each LIF layer) is
scored so that cascade-law quantities can be computed downstream. The decoder
time constant is a readout choice made after the spikes exist, so one network
run is scored at every requested `decoder_tau` rather than re-simulated. R5
generalises this: a run is scored with every readout in `Spec.readouts`, and
the encoder can be temporal or delta instead of Poisson (`ENCODERS`).

Results are appended row by row and a grid resumes from whatever its CSV
already holds, so an interrupted sweep loses at most the runs in flight.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
from neuroflux_core import (
    DeltaEncoder,
    LIFConfig,
    Network,
    PoissonRateEncoder,
    STDPConfig,
    TemporalEncoder,
    calibrate_delta_threshold,
    calibrate_encoder_rate,
    calibrate_gain,
)
from neuroflux_core.encoders.base import BaseEncoder
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_core.metrics.reconstruction import affine_corrected_snr, affine_fit
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_datasets.signals import DEFAULT_CACHE_DIR, load_reference_beats
from neuroflux_datasets.synthetic import (
    add_interference,
    generate_ecg,
    generate_eeg,
    generate_sine,
)
from scipy.signal import butter, filtfilt, lfilter  # type: ignore[import-untyped]

FS = 250.0
DT = 1000.0 / FS
MAX_RATE_HZ = 100.0
DENSE_RATE_HZ = 40.0
DURATION_S = 10.0
CALIBRATION_S = 2.0
# Below this target a 2 s window holds too few spikes to estimate a rate, so
# calibration uses the whole signal instead.
SPARSE_CALIBRATION_HZ = 10.0
# After a spike the LIF kernel skips integration while `refractory_time > 0`,
# so a neuron can fire at most once every `1 + ceil(t_ref / dt)` steps: 125 Hz
# at dt = 4 ms, which the probe measured directly.
RATE_LIMIT_HZ = 1000.0 / (DT * (1 + math.ceil(LIFConfig().t_ref / DT)))
# A neuron at this fraction of the limit is pinned by its refractory period
# and carries no amplitude information.
SATURATED_RATE_FRACTION = 0.9

# Degeneracy verdicts, fixed before any R3 grid was run.
#   silent     the output is constant (`is_degenerate`) — nothing fired, or
#              too little to move the readout;
#   saturated  at least half the output neurons fire at >= 90% of the rate
#              limit. A pinned population is *not* constant: its readout
#              ripples at the refractory period, so its spread stays large and
#              only the rates reveal it (shipped [256,16,8] probe: output std
#              0.17 of the signal's, affine SNR 0.0006 dB);
#   ok         anything else, including a poor but live reconstruction.
SATURATED_POPULATION_FRACTION = 0.5

CLEAN = math.inf

SIGNALS: tuple[str, ...] = ("synthetic-ecg", "mit-bih", "eeg", "sine")
OPERATING_POINTS: tuple[str, ...] = ("shipped", "controlled")


_LATE_FIELDS: frozenset[str] = frozenset(
    {
        "calibrate_at_snr_db",
        "encoder_range",
        "fan_in",
        "encoder",
        "readouts",
        "weight_bits",
        "weight_sigma",
        "weight_stuck",
    }
)

# Encoders (R5). Every R3 and R4b grid used Poisson, the default.
#   poisson   rate code; calibrated to its target mean rate by its ceiling;
#   temporal  time-to-first-spike: one spike per channel per window, earlier
#             for larger values. The window is the rate: 24 ms (6 steps,
#             41.7 Hz, six latency levels) for a 40 Hz target;
#   delta     `DeltaEncoder` in absolute mode, its threshold calibrated to the
#             target rate on the calibration drive. Its rate follows
#             `|dx/dt|`, so a rate readout sees the slope, not the signal.
# Temporal and delta encoders are deterministic, and a tiled drive gives every
# input channel the same spikes.
ENCODERS: tuple[str, ...] = ("poisson", "temporal", "delta")

# Readouts (R5), each a (kind, parameter) pair applied to every stage.
#   population  `PopulationDecoder`, parameter its time constant in ms;
#   spike-rate  `SpikeRateDecoder`, parameter its trailing window in ms;
#   latency     temporal encoder only, parameter ignored. See `latency_readout`.
READOUT_KINDS: tuple[str, ...] = ("population", "spike-rate", "latency")
# A deterministic encoder has no ceiling parameter; its stage is read against
# the most a channel can fire, once per step.
STEP_RATE_HZ = 1000.0 / DT
# Lags searched for `snr_best_lag_db`, which separates a readout's delay from
# its noise. The longest causal readout in R5 trails by about 40 ms; a longer
# search starts matching narrowband signals a whole cycle late (EEG alpha has a
# 100 ms period), which is why the bound is short and positive scales only.
MAX_LAG_MS = 64.0

# How the encoder's [v_min, v_max] is chosen (R4b, roadmap next action 14).
#   None        the drive's own minimum and maximum — every R3 grid;
#   percentile  its 0.1st and 99.9th percentiles, which ignore the extreme
#               noise samples but clip the tips of the tallest QRS peaks;
#   lowpass     the extremes of a zero-phase 40 Hz low-pass copy, which drops
#               most broadband noise but keeps the ECG's band;
#   oracle      the clean signal's extremes — not deployable, an upper bound.
ENCODER_RANGES: tuple[str, ...] = ("percentile", "lowpass", "oracle")
RANGE_PERCENTILE = 0.1
RANGE_LOWPASS_HZ = 40.0


@dataclass(frozen=True)
class Spec:
    """One network run. Every field is part of its identity for resuming."""

    experiment: str
    signal: str
    unit: str
    seed: int
    input_snr_db: float = CLEAN
    interference: str = "gaussian"
    layers: tuple[int, ...] = (32, 16, 8)
    operating_point: str = "controlled"
    epochs: int = 0
    stdp_tau: float = 20.0
    decoder_taus: tuple[float, ...] = (20.0,)
    # Controlled only: one target per LIF layer; None holds every layer dense.
    layer_rates_hz: tuple[float, ...] | None = None
    encoder_rate_hz: float = DENSE_RATE_HZ
    # Shipped only: the single gain every layer shares.
    g_syn: float = 30.0
    duration_s: float = DURATION_S
    # Diagnostic only: take the operating point — encoder range, encoder rate
    # calibration and layer gains — from the same seed's drive at this input
    # SNR, while the network itself runs on `input_snr_db`. A noiseless run
    # calibrated at s dB measures the stages' intrinsic SNRs *at the operating
    # point a noisy s dB input sets*, which is what the cascade law needs if
    # that operating point moves with the noise.
    calibrate_at_snr_db: float | None = None
    # R4b: see ENCODER_RANGES.
    encoder_range: str | None = None
    # R4b: exact number of non-zero inputs per first-layer neuron (M4's scheme);
    # None keeps the dense initialisation.
    fan_in: int | None = None
    # R5: see ENCODERS; None is Poisson.
    encoder: str | None = None
    # R5: see READOUT_KINDS. When set, replaces `decoder_taus`; None reads the
    # output with a population decoder at each of `decoder_taus`.
    readouts: tuple[tuple[str, float], ...] | None = None
    # H1: device non-idealities applied to every weight matrix once it is built
    # and before calibration; see `perturb_weights`. None leaves weights exact.
    weight_bits: int | None = None
    weight_sigma: float | None = None
    weight_stuck: float | None = None

    def key(self) -> str:
        """
        A stable identifier, written into every row, for resuming a grid.

        Fields added after the first R3 grid ran appear only when they differ
        from their default, so rows written before they existed keep their key.
        """
        return "|".join(
            f"{name}={value}"
            for name, value in asdict(self).items()
            if not (name in _LATE_FIELDS and value is None)
        )

    @property
    def init_scheme(self) -> Literal["uniform", "scaled"]:
        """Shipped networks start uniform, as `cli.py` builds them."""
        return "uniform" if self.operating_point == "shipped" else "scaled"

    @property
    def readout_list(self) -> tuple[tuple[str, float], ...]:
        """Every (kind, parameter) this run is scored with, in row order."""
        if self.readouts is not None:
            return self.readouts
        return tuple(("population", tau) for tau in self.decoder_taus)


@dataclass(frozen=True)
class Row:
    """One run scored at one decoder time constant."""

    spec_key: str
    experiment: str
    signal: str
    unit: str
    seed: int
    operating_point: str
    interference: str
    layers: str
    init_scheme: str
    epochs: int
    stdp_tau: float
    decoder_tau: float
    input_snr_nominal: float
    input_snr_db: float
    input_affine_db: float
    calibrate_at_snr_db: float
    encoder_range: str
    fan_in: int
    encoder: str
    decoder: str
    g_syn: float
    encoder_max_hz: float
    delta_threshold: float
    layer_rate_targets_hz: str
    layer_gains: str
    snr_raw_db: float
    snr_affine_db: float
    affine_scale: float
    snr_best_lag_db: float
    best_lag_ms: float
    verdict: str
    degenerate: bool
    out_std_rel: float
    stage_affine_db: str
    latency_delays: str
    encoder_rate_hz: float
    layer_rates_hz: str
    out_rate_hz: float
    silent_frac_out: float
    saturated_frac_out: float
    silent_frac_all: float
    qrs_f1: float
    qrs_sensitivity: float
    qrs_ppv: float
    qrs_jitter_ms: float
    qrs_f1_input: float
    wall_s: float


def _join(values: Sequence[float]) -> str:
    return ";".join(f"{v:.6g}" for v in values)


def split_floats(field: str) -> list[float]:
    """Parse a `;`-joined per-stage field back into floats."""
    return [float(v) for v in field.split(";")] if field else []


def load_signal(
    signal: str, unit: str, seed: int, duration_s: float, cache_dir: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    The clean window for one (signal, unit, seed), and its beats if annotated.

    MIT-BIH takes a different artifact-free window per seed. Synthetic ECG and
    the sine are deterministic, so their seeds replicate only the noise, the
    weights and the spikes; EEG draws new band phases per seed.
    """
    if signal == "mit-bih":
        clean, beats = load_reference_beats(
            unit, duration_s=duration_s, fs=FS, window=seed, cache_dir=cache_dir
        )
        return np.asarray(clean, dtype=np.float64), beats
    if signal == "synthetic-ecg":
        return generate_ecg(duration_s=duration_s, fs=FS), None
    if signal == "eeg":
        phase_rng = np.random.default_rng(np.random.SeedSequence([seed, 20_000]))
        return generate_eeg(duration_s=duration_s, fs=FS, rng=phase_rng), None
    if signal == "sine":
        return generate_sine(duration_s=duration_s, fs=FS), None
    raise ValueError(f"Unknown signal {signal!r}; expected one of {SIGNALS}.")


def _streams(seed: int) -> tuple[np.random.Generator, ...]:
    """Independent noise, encoder and calibration streams for one seed."""
    return tuple(
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )


def readout(
    spikes: np.ndarray,
    v_min: float,
    v_max: float,
    tau: float,
    max_rate_hz: float,
) -> np.ndarray:
    """
    `PopulationDecoder.decode` applied to a whole spike raster at once.

    The exponential trace `s_t = s_{t-1} e^{-dt/tau} + x_t` is a one-pole IIR
    filter, so `lfilter` computes it for every neuron in one call.
    """
    decay = math.exp(-DT / tau)
    traces = lfilter([1.0], [1.0, -decay], spikes.astype(np.float64), axis=0)
    clipped = np.clip(traces / (tau / 1000.0) / max_rate_hz, 0.0, 1.0)
    return np.asarray((v_min + clipped * (v_max - v_min)).mean(axis=1))


def spike_rate_readout(
    spikes: np.ndarray,
    v_min: float,
    v_max: float,
    window_ms: float,
    max_rate_hz: float,
) -> np.ndarray:
    """
    `SpikeRateDecoder.decode` applied to a whole spike raster at once.

    The decoder keeps the trailing `floor(window_ms / dt)` steps (at least
    one), divides the population's count by the time they span and clips the
    population mean rate, not each neuron's, at `max_rate_hz`.
    """
    steps = max(1, int(math.floor(window_ms / DT + 1e-9)))
    n_steps, n_neurons = spikes.shape
    cumulative = np.concatenate([[0.0], np.cumsum(spikes.sum(axis=1, dtype=float))])
    end = np.arange(1, n_steps + 1)
    start = np.maximum(0, end - steps)
    counts = cumulative[end] - cumulative[start]
    elapsed_ms = (end - start) * DT
    rate = counts / elapsed_ms * 1000.0 / max(n_neurons, 1)
    normalized = np.minimum(rate / max_rate_hz, 1.0)
    return np.asarray(v_min + normalized * (v_max - v_min))


def temporal_window_steps(rate_hz: float) -> int:
    """Steps per temporal-encoder window for a target rate: one spike per window."""
    return max(2, round(1000.0 / (rate_hz * DT)))


def latency_readout(
    spikes: np.ndarray, encoder_spikes: np.ndarray, window_steps: int
) -> tuple[np.ndarray, int]:
    """
    A time-to-first-spike code read back out of any stage, causally.

    The encoder samples the drive at the start of each window and every
    channel fires once inside it, earlier for larger values. A later stage
    answers each volley after a delay, so its windows are shifted by that
    delay: the lag in `[0, window_steps)` at which the stage's pooled spike
    count correlates best with the encoder's. The delay is found from spikes
    alone — the clean signal is never consulted.

    Each shifted window is read as the population's mean first-spike step,
    with a neuron that stays silent counted at `window_steps`, negated so that
    earlier means larger. That value is known only once its window has closed,
    so it is held from then until the next window closes.

    Returns:
        The readout, one value per step, and the delay in steps.
    """
    n_steps = spikes.shape[0]
    pooled_encoder = encoder_spikes.sum(axis=1, dtype=float)
    pooled = spikes.sum(axis=1, dtype=float)
    correlation = [
        float(np.dot(pooled_encoder[: n_steps - lag], pooled[lag:]))
        for lag in range(window_steps)
    ]
    delay = int(np.argmax(correlation))
    n_windows = (n_steps - delay) // window_steps
    values = np.empty(max(n_windows, 1))
    values[0] = 0.0
    for k in range(n_windows):
        block = spikes[delay + k * window_steps : delay + (k + 1) * window_steps]
        fired = block.any(axis=0)
        first = np.where(fired, block.argmax(axis=0), window_steps)
        values[k] = -float(first.mean())
    out = np.full(n_steps, values[0])
    for k in range(n_windows):
        start = delay + (k + 1) * window_steps
        out[start : start + window_steps] = values[k]
    if n_windows:
        tail = delay + (n_windows + 1) * window_steps
        out[tail:] = values[n_windows - 1]
    return out, delay


def best_lag_snr(clean: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    """
    Affine-corrected SNR with the output advanced by its best lag, and the lag.

    Every causal readout trails the signal, and at high SNR a few tens of
    milliseconds cost as much as the noise does; this separates the two. Lags
    run from 0 to `MAX_LAG_MS`, every lag is scored on the same samples (the
    first `T - MAX_LAG_MS` of the clean signal), and only fits with a positive
    scale count, so an inverted or half-cycle-late match cannot win. Taking the
    best of 17 neighbouring lags is slightly optimistic; the same search is
    applied to every readout.
    """
    max_lag = int(MAX_LAG_MS / DT)
    n = clean.shape[0] - max_lag
    reference = clean[:n]
    best, best_lag = -math.inf, 0
    for lag in range(max_lag + 1):
        segment = output[lag : lag + n]
        if is_degenerate(segment):
            continue
        scale, _ = affine_fit(reference, segment)
        if scale <= 0.0:
            continue
        value = affine_corrected_snr(reference, segment)
        if value > best:
            best, best_lag = value, lag
    if math.isinf(best):
        return math.nan, math.nan
    return best, best_lag * DT


def encoder_bounds(
    drive: np.ndarray, clean: np.ndarray, method: str | None
) -> tuple[float, float]:
    """The encoder's `[v_min, v_max]` from a 1-D drive, by `ENCODER_RANGES` method."""
    if method is None:
        low, high = float(np.min(drive)), float(np.max(drive))
    elif method == "percentile":
        low = float(np.percentile(drive, RANGE_PERCENTILE))
        high = float(np.percentile(drive, 100.0 - RANGE_PERCENTILE))
    elif method == "lowpass":
        b, a = butter(4, RANGE_LOWPASS_HZ / (FS / 2.0))
        smooth = filtfilt(b, a, drive)
        low, high = float(np.min(smooth)), float(np.max(smooth))
    elif method == "oracle":
        low, high = float(np.min(clean)), float(np.max(clean))
    else:
        raise ValueError(
            f"Unknown encoder range {method!r}; expected {ENCODER_RANGES}."
        )
    if high <= low:
        high = low + 1.0
    return low, high


def fixed_fan_in_weights(
    n_pre: int, n_post: int, fan_in: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Uniform[0, 1] weights with exactly `fan_in` non-zero inputs per neuron.

    Copied from `scripts/m4_decorrelation/density_sweep.py` so R4b's fan-in runs
    use M4's construction; the scripts tree is not an importable package.
    """
    if not 1 <= fan_in <= n_pre:
        raise ValueError(f"fan_in must be in [1, {n_pre}], got {fan_in}")
    weights = np.zeros((n_pre, n_post), dtype=np.float32)
    for post in range(n_post):
        chosen = rng.choice(n_pre, size=fan_in, replace=False)
        weights[chosen, post] = rng.uniform(0.0, 1.0, size=fan_in)
    return weights


def perturb_weights(
    weights: np.ndarray,
    seed: int,
    layer: int,
    bits: int | None = None,
    sigma: float | None = None,
    stuck: float | None = None,
) -> np.ndarray:
    """
    A weight matrix as a non-ideal device array would realise it (H1 pilot A).

    Applied in the order an array imposes them. `bits` quantizes to `2**bits`
    evenly spaced levels on `[0, max(w)]`, the matrix's own range, as
    programming maps it. `sigma` multiplies each weight by `exp(sigma * z)`,
    `z ~ N(0, 1)`: lognormal device-to-device variation. `stuck` sets each
    weight to zero with that probability: an open device. Zeros stay zero, so
    a fixed fan-in keeps its mask.

    The variation and fault draws have their own streams, fixed by `seed` and
    `layer` and drawn whatever the level, so successive levels perturb the same
    devices and paired comparisons across levels are paired by device too.
    """
    if bits is not None and bits < 1:
        raise ValueError(f"bits must be at least 1, got {bits}")
    if sigma is not None and sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if stuck is not None and not 0.0 <= stuck < 1.0:
        raise ValueError(f"stuck must be in [0, 1), got {stuck}")
    out = np.asarray(weights, dtype=np.float64).copy()
    if bits is not None:
        top = float(out.max())
        if top > 0.0:
            levels = 2**bits - 1
            out = np.round(out / top * levels) / levels * top
    if sigma is not None:
        z = np.random.default_rng(
            np.random.SeedSequence([seed, 40_000, layer])
        ).standard_normal(out.shape)
        out = out * np.exp(sigma * z)
    if stuck is not None:
        u = np.random.default_rng(np.random.SeedSequence([seed, 50_000, layer])).random(
            out.shape
        )
        out = np.where(u < stuck, 0.0, out)
    return out.astype(np.float32)


def _calibration_spikes(
    encoder: BaseEncoder, drive: np.ndarray, seconds: float
) -> np.ndarray:
    steps = min(drive.shape[0], int(seconds * FS))
    return np.stack([encoder.encode(drive[t], DT) for t in range(steps)], axis=0)


def _make_encoder(
    name: str,
    v_min: float,
    v_max: float,
    encoder_max_hz: float,
    threshold: float,
    window_steps: int,
    rng: np.random.Generator,
) -> BaseEncoder:
    if name == "poisson":
        return PoissonRateEncoder(
            v_min=v_min, v_max=v_max, max_firing_rate_hz=encoder_max_hz, rng=rng
        )
    if name == "temporal":
        return TemporalEncoder(v_min, v_max, t_window_ms=window_steps * DT)
    return DeltaEncoder(threshold, mode="absolute")


# (clean, noisy, reference beats or None, {(kind, parameter): output signal})
OutputHook = Callable[
    [np.ndarray, np.ndarray, "np.ndarray | None", dict[tuple[str, float], np.ndarray]],
    None,
]


def run_spec(
    spec: Spec,
    cache_dir: str = DEFAULT_CACHE_DIR,
    network_hook: Callable[[FeedForwardNetwork], None] | None = None,
    output_hook: OutputHook | None = None,
) -> list[Row]:
    """
    Build, calibrate, train, run and score one spec with every readout.

    `network_hook`, if given, is called with the network after the evaluation
    run, for diagnostics that need its weights. It must not modify the network;
    the rows are computed from spikes already captured. `output_hook`, if
    given, receives every readout's output signal, for scoring the rows do not
    carry.
    """
    started = time.perf_counter()
    if spec.operating_point not in OPERATING_POINTS:
        raise ValueError(f"Unknown operating point {spec.operating_point!r}.")
    controlled = spec.operating_point == "controlled"
    n_groups = len(spec.layers) - 1
    encoder_name = spec.encoder or "poisson"
    if encoder_name not in ENCODERS:
        raise ValueError(f"Unknown encoder {encoder_name!r}; expected {ENCODERS}.")
    if encoder_name != "poisson" and not controlled:
        raise ValueError("Temporal and delta encoders need the controlled point.")
    for kind, _ in spec.readout_list:
        if kind not in READOUT_KINDS:
            raise ValueError(f"Unknown readout {kind!r}; expected {READOUT_KINDS}.")
        if kind == "latency" and encoder_name != "temporal":
            raise ValueError("The latency readout needs the temporal encoder.")

    noise_rng, encoder_rng, calib_rng = _streams(spec.seed)
    clean, beats = load_signal(
        spec.signal, spec.unit, spec.seed, spec.duration_s, cache_dir
    )
    noisy = (
        clean.copy()
        if math.isinf(spec.input_snr_db)
        else add_interference(
            clean,
            spec.input_snr_db,
            kind=spec.interference,  # type: ignore[arg-type]
            fs=FS,
            rng=noise_rng,
        )
    )
    drive = np.tile(noisy, (spec.layers[0], 1)).T
    # The drive the operating point is set from. Normally the run's own; for
    # the diagnostic, the same seed's interference at another SNR, drawn from
    # a fresh copy of the noise stream so its shape matches.
    if spec.calibrate_at_snr_db is None:
        calibration_drive = drive
    else:
        reference = (
            clean.copy()
            if math.isinf(spec.calibrate_at_snr_db)
            else add_interference(
                clean,
                spec.calibrate_at_snr_db,
                kind=spec.interference,  # type: ignore[arg-type]
                fs=FS,
                rng=_streams(spec.seed)[0],
            )
        )
        calibration_drive = np.tile(reference, (spec.layers[0], 1)).T
    v_min, v_max = encoder_bounds(calibration_drive[:, 0], clean, spec.encoder_range)

    targets = (
        list(spec.layer_rates_hz)
        if spec.layer_rates_hz is not None
        else [DENSE_RATE_HZ] * n_groups
    )
    if controlled and len(targets) != n_groups:
        raise ValueError(
            f"{len(targets)} layer rate targets for {n_groups} neuron groups."
        )
    encoder_max_hz = math.nan
    threshold = math.nan
    window_steps = temporal_window_steps(spec.encoder_rate_hz)
    if encoder_name == "poisson":
        encoder_max_hz = (
            calibrate_encoder_rate(
                calibration_drive, v_min, v_max, DT, target_rate_hz=spec.encoder_rate_hz
            )
            if controlled
            else MAX_RATE_HZ
        )
    elif encoder_name == "delta":
        threshold = calibrate_delta_threshold(
            calibration_drive[:, 0], DT, target_rate_hz=spec.encoder_rate_hz
        )

    net = Network.feedforward(
        layers=list(spec.layers),
        neuron=LIFConfig() if controlled else LIFConfig(g_syn=spec.g_syn),
        plasticity=STDPConfig(tau_plus=spec.stdp_tau, tau_minus=spec.stdp_tau),
        seed=spec.seed,
        init_scheme=spec.init_scheme,
    )
    if spec.fan_in is not None:
        weights = fixed_fan_in_weights(
            spec.layers[0],
            spec.layers[1],
            spec.fan_in,
            np.random.default_rng(np.random.SeedSequence([spec.seed, 30_000])),
        )
        first = net.synapses[0]
        first.weights = weights.copy()
        # `reset_state` restores from this copy, so it must carry the new matrix.
        first.initial_weights_copy = weights.copy()
    if (
        spec.weight_bits is not None
        or spec.weight_sigma is not None
        or spec.weight_stuck is not None
    ):
        for layer, synapse in enumerate(net.synapses):
            realised = perturb_weights(
                synapse.weights,
                spec.seed,
                layer,
                bits=spec.weight_bits,
                sigma=spec.weight_sigma,
                stuck=spec.weight_stuck,
            )
            synapse.weights = realised.copy()
            synapse.initial_weights_copy = realised.copy()

    def new_encoder(rng: np.random.Generator) -> BaseEncoder:
        return _make_encoder(
            encoder_name, v_min, v_max, encoder_max_hz, threshold, window_steps, rng
        )

    encoder = new_encoder(encoder_rng)
    if isinstance(encoder, PoissonRateEncoder):
        poisson = encoder
        stream_state = poisson.rng.bit_generator.state

        def rewind() -> None:
            poisson.rng.bit_generator.state = stream_state

    else:
        rewind = encoder.reset_state

    if controlled:
        sparse = min(targets) < SPARSE_CALIBRATION_HZ
        calib_spikes = _calibration_spikes(
            new_encoder(calib_rng),
            calibration_drive,
            spec.duration_s if sparse else CALIBRATION_S,
        )
        tol_hz = min(0.5, 0.05 * min(targets))

        def calibrate() -> None:
            calibrate_gain(
                net,
                calib_spikes,
                dt=DT,
                target_rate_hz=targets,
                tol_hz=tol_hz,
                max_iter=22,
            )

        calibrate()

    n_steps = drive.shape[0]
    if spec.epochs:
        # Train on the encoder stream the evaluation will also see, then rewind
        # it, so epochs = 0 and epochs = N are scored on identical spikes.
        for _ in range(spec.epochs):
            net.reset_transient_state()
            for t in range(n_steps):
                net.step(encoder.encode(drive[t], DT), DT, training=True)
            rewind()
        if controlled:
            calibrate()

    net.reset_transient_state()
    rewind()
    widths = list(spec.layers)
    rasters = [np.zeros((n_steps, w), dtype=bool) for w in widths]
    for t in range(n_steps):
        for idx, spikes in enumerate(net.step(encoder.encode(drive[t], DT), DT, False)):
            rasters[idx][t] = spikes

    if network_hook is not None:
        network_hook(net)

    rates = [float(r.mean() * 1000.0 / DT) for r in rasters]
    per_neuron_out = rasters[-1].mean(axis=0)
    per_neuron_rate = per_neuron_out * 1000.0 / DT
    saturated_frac = float(
        np.mean(per_neuron_rate >= SATURATED_RATE_FRACTION * RATE_LIMIT_HZ)
    )
    silent_frac = float(np.mean(per_neuron_out == 0.0))
    lif_counts = np.concatenate([r.sum(axis=0) for r in rasters[1:]])

    input_db = CLEAN if math.isinf(spec.input_snr_db) else calculate_snr(clean, noisy)
    input_affine = (
        CLEAN if math.isinf(spec.input_snr_db) else affine_corrected_snr(clean, noisy)
    )
    qrs_input = math.nan
    if beats is not None:
        qrs_input = qrs_score(beats, noisy, FS).f1

    clean_std = float(np.std(clean))
    encoder_stage_hz = encoder_max_hz if encoder_name == "poisson" else STEP_RATE_HZ
    rows: list[Row] = []
    outputs: dict[tuple[str, float], np.ndarray] = {}
    for kind, parameter in spec.readout_list:
        delays: list[int] = []
        stages = []
        for idx, raster in enumerate(rasters):
            stage_hz = encoder_stage_hz if idx == 0 else MAX_RATE_HZ
            if kind == "population":
                stages.append(readout(raster, v_min, v_max, parameter, stage_hz))
            elif kind == "spike-rate":
                stages.append(
                    spike_rate_readout(raster, v_min, v_max, parameter, stage_hz)
                )
            else:
                signal, delay = latency_readout(raster, rasters[0], window_steps)
                stages.append(signal)
                delays.append(delay)
        output = stages[-1]
        outputs[(kind, parameter)] = output
        degenerate = is_degenerate(output)
        if saturated_frac >= SATURATED_POPULATION_FRACTION:
            verdict = "saturated"
        elif degenerate:
            verdict = "silent"
        else:
            verdict = "ok"
        stage_db = [
            math.nan if is_degenerate(s) else affine_corrected_snr(clean, s)
            for s in stages
        ]
        best_db, best_lag_ms = (
            (math.nan, math.nan) if degenerate else best_lag_snr(clean, output)
        )
        f1 = se = ppv = jitter = math.nan
        if beats is not None:
            score = qrs_score(beats, output, FS)
            f1, se, ppv, jitter = (
                score.f1,
                score.sensitivity,
                score.ppv,
                score.rms_jitter_ms,
            )
        rows.append(
            Row(
                spec_key=spec.key(),
                experiment=spec.experiment,
                signal=spec.signal,
                unit=spec.unit,
                seed=spec.seed,
                operating_point=spec.operating_point,
                interference=spec.interference,
                layers="-".join(str(w) for w in spec.layers),
                init_scheme=spec.init_scheme,
                epochs=spec.epochs,
                stdp_tau=spec.stdp_tau,
                decoder_tau=math.nan if kind == "latency" else parameter,
                input_snr_nominal=spec.input_snr_db,
                input_snr_db=input_db,
                input_affine_db=input_affine,
                calibrate_at_snr_db=(
                    math.nan
                    if spec.calibrate_at_snr_db is None
                    else spec.calibrate_at_snr_db
                ),
                encoder_range=spec.encoder_range or "minmax",
                fan_in=0 if spec.fan_in is None else spec.fan_in,
                encoder=encoder_name,
                decoder=kind,
                g_syn=math.nan if controlled else spec.g_syn,
                encoder_max_hz=encoder_max_hz,
                delta_threshold=threshold,
                layer_rate_targets_hz=_join(targets) if controlled else "",
                layer_gains=_join(net.layer_gains),
                snr_raw_db=math.nan if degenerate else calculate_snr(clean, output),
                snr_affine_db=stage_db[-1],
                affine_scale=(math.nan if degenerate else affine_fit(clean, output)[0]),
                snr_best_lag_db=best_db,
                best_lag_ms=best_lag_ms,
                verdict=verdict,
                degenerate=degenerate,
                out_std_rel=float(np.std(output)) / clean_std if clean_std else 0.0,
                stage_affine_db=_join(stage_db),
                latency_delays=";".join(str(d) for d in delays),
                encoder_rate_hz=rates[0],
                layer_rates_hz=_join(rates[1:]),
                out_rate_hz=rates[-1],
                silent_frac_out=silent_frac,
                saturated_frac_out=saturated_frac,
                silent_frac_all=float(np.mean(lif_counts == 0)),
                qrs_f1=f1,
                qrs_sensitivity=se,
                qrs_ppv=ppv,
                qrs_jitter_ms=jitter,
                qrs_f1_input=qrs_input,
                wall_s=time.perf_counter() - started,
            )
        )
    if output_hook is not None:
        output_hook(clean, noisy, beats, outputs)
    return rows


def completed_keys(path: Path) -> set[str]:
    """Spec keys already present in a results CSV."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["spec_key"] for row in csv.DictReader(handle)}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Every row of a results CSV, as strings."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_grid(
    specs: list[Spec],
    out: Path,
    workers: int | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    task: Callable[[Spec, str], Sequence[Any]] = run_spec,
    row_type: type = Row,
) -> None:
    """
    Run every spec not already in `out`, appending rows as runs finish.

    `task` maps a spec and the cache directory to its rows, which must be
    dataclasses of `row_type` carrying a `spec_key` field; it must be a
    module-level function so the process pool can pickle it.

    Raises `SystemExit` after the pool drains if any run failed, having kept
    every row that succeeded — the loud-failure contract `cli.py` follows.
    """
    keys = [s.key() for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("The grid contains duplicate specs.")
    done = completed_keys(out)
    todo = [s for s in specs if s.key() not in done]
    print(
        f"{len(specs)} runs in grid, {len(specs) - len(todo)} already in {out}, "
        f"{len(todo)} to run."
    )
    if not todo:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    header = [f.name for f in fields(row_type)]
    if not new_file:
        with out.open(newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), [])
        if existing != header:
            raise SystemExit(
                f"{out} was written with different columns than this code "
                f"produces; appending would corrupt it. Write to a new file."
            )
    failures: list[tuple[Spec, str]] = []
    started = time.perf_counter()
    with (
        out.open("a", newline="", encoding="utf-8") as handle,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        writer = csv.DictWriter(handle, fieldnames=header)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(task, spec, cache_dir): spec for spec in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            try:
                rows = future.result()
            except Exception as exc:  # noqa: BLE001 — reported and re-raised below
                failures.append((spec, repr(exc)))
                continue
            writer.writerows(asdict(r) for r in rows)
            handle.flush()
            if count % max(1, len(todo) // 50) == 0 or count == len(todo):
                elapsed = time.perf_counter() - started
                remaining = elapsed / count * (len(todo) - count)
                print(
                    f"  {count}/{len(todo)} runs, {elapsed / 60:.1f} min elapsed, "
                    f"~{remaining / 60:.1f} min left, {len(failures)} failed",
                    flush=True,
                )

    if failures:
        for spec, message in failures[:20]:
            print(f"FAILED {spec.key()}: {message}")
        raise SystemExit(f"{len(failures)} of {len(todo)} runs failed.")
