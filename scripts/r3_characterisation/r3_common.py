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
run is scored at every requested `decoder_tau` rather than re-simulated.

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
from typing import Literal

import numpy as np
from neuroflux_core import (
    LIFConfig,
    Network,
    PoissonRateEncoder,
    STDPConfig,
    calibrate_encoder_rate,
    calibrate_gain,
)
from neuroflux_core.metrics.degeneracy import is_degenerate
from neuroflux_core.metrics.qrs import qrs_score
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
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
    {"calibrate_at_snr_db", "encoder_range", "fan_in"}
)

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
    g_syn: float
    encoder_max_hz: float
    layer_rate_targets_hz: str
    layer_gains: str
    snr_raw_db: float
    snr_affine_db: float
    verdict: str
    degenerate: bool
    out_std_rel: float
    stage_affine_db: str
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


def _calibration_spikes(
    drive: np.ndarray,
    v_min: float,
    v_max: float,
    encoder_max_hz: float,
    rng: np.random.Generator,
    seconds: float,
) -> np.ndarray:
    encoder = PoissonRateEncoder(
        v_min=v_min, v_max=v_max, max_firing_rate_hz=encoder_max_hz, rng=rng
    )
    steps = min(drive.shape[0], int(seconds * FS))
    return np.stack([encoder.encode(drive[t], DT) for t in range(steps)], axis=0)


def run_spec(
    spec: Spec,
    cache_dir: str = DEFAULT_CACHE_DIR,
    network_hook: Callable[[FeedForwardNetwork], None] | None = None,
) -> list[Row]:
    """
    Build, calibrate, train, run and score one spec at every decoder tau.

    `network_hook`, if given, is called with the network after the evaluation
    run, for diagnostics that need its weights. It must not modify the network;
    the rows are computed from spikes already captured.
    """
    started = time.perf_counter()
    if spec.operating_point not in OPERATING_POINTS:
        raise ValueError(f"Unknown operating point {spec.operating_point!r}.")
    controlled = spec.operating_point == "controlled"
    n_groups = len(spec.layers) - 1

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
    encoder_max_hz = (
        calibrate_encoder_rate(
            calibration_drive, v_min, v_max, DT, target_rate_hz=spec.encoder_rate_hz
        )
        if controlled
        else MAX_RATE_HZ
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
    encoder = PoissonRateEncoder(
        v_min=v_min, v_max=v_max, max_firing_rate_hz=encoder_max_hz, rng=encoder_rng
    )

    if controlled:
        sparse = min(targets) < SPARSE_CALIBRATION_HZ
        calib_spikes = _calibration_spikes(
            calibration_drive,
            v_min,
            v_max,
            encoder_max_hz,
            calib_rng,
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
        stream_state = encoder.rng.bit_generator.state
        for _ in range(spec.epochs):
            net.reset_transient_state()
            for t in range(n_steps):
                net.step(encoder.encode(drive[t], DT), DT, training=True)
            encoder.rng.bit_generator.state = stream_state
        if controlled:
            calibrate()

    net.reset_transient_state()
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
    rows: list[Row] = []
    for tau in spec.decoder_taus:
        stages = [
            readout(
                raster,
                v_min,
                v_max,
                tau,
                encoder_max_hz if idx == 0 else MAX_RATE_HZ,
            )
            for idx, raster in enumerate(rasters)
        ]
        output = stages[-1]
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
                decoder_tau=tau,
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
                g_syn=math.nan if controlled else spec.g_syn,
                encoder_max_hz=encoder_max_hz,
                layer_rate_targets_hz=_join(targets) if controlled else "",
                layer_gains=_join(net.layer_gains),
                snr_raw_db=math.nan if degenerate else calculate_snr(clean, output),
                snr_affine_db=stage_db[-1],
                verdict=verdict,
                degenerate=degenerate,
                out_std_rel=float(np.std(output)) / clean_std if clean_std else 0.0,
                stage_affine_db=_join(stage_db),
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
) -> None:
    """
    Run every spec not already in `out`, appending rows as runs finish.

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
    header = [f.name for f in fields(Row)]
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
        futures = {pool.submit(run_spec, spec, cache_dir): spec for spec in todo}
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
