"""Command Line Interface for NeuroFlux Experiments."""

import argparse
import concurrent.futures
import itertools
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.config import LIFConfig, STDPConfig
from neuroflux_core.metrics.qrs import QrsScore
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_core.network.feedforward import FeedForwardNetwork, Network
from neuroflux_core.network.simulator import Simulator
from pydantic import BaseModel, ConfigDict, Field

from neuroflux_datasets.synthetic import SignalGenerator


class Phase2Config(BaseModel):
    """Pydantic schema for Phase 2 validation sweep configurations."""

    model_config = ConfigDict(extra="forbid")

    signal_types: list[Literal["ecg", "eeg", "sine"]]
    flicker_frequencies: list[float]
    input_snrs: list[float]
    training_epochs: list[int]
    stdp_taus: list[float]
    flicker_amplitude: float = 0.5
    seeds_per_config: int = 10
    duration_s: float = 1.0
    network_layers: list[int] = Field(default_factory=lambda: [32, 16, 8])


class Phase3Config(BaseModel):
    """Pydantic schema for Phase 3 MIT-BIH sweep configurations."""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["mit-bih"] = "mit-bih"
    record_ids: list[str]
    flicker_frequencies: list[float]
    input_snrs: list[float]
    training_epochs: list[int]
    stdp_taus: list[float]
    flicker_amplitude: float = 0.5
    seeds_per_config: int = 10
    duration_s: float = 10.0
    cache_dir: str = "data/mit_bih"
    network_layers: list[int] = Field(default_factory=lambda: [32, 16, 8])


@dataclass
class RunResult:
    signal_type: str
    flicker_freq: float
    input_snr: float
    epochs: int
    stdp_tau: float
    seed: int
    delta_snr: float
    convergence_speed: int
    dataset: str = "synthetic"
    record_id: str | None = None
    rpeak_integrity: float | None = None
    # R1: annotation-referenced QRS scores, replacing the saturated
    # `rpeak_integrity`. MIT-BIH only; `qrs_jitter_ms` is None when nothing
    # matched. See docs/r1_findings.md.
    qrs_f1: float | None = None
    qrs_sensitivity: float | None = None
    qrs_ppv: float | None = None
    qrs_jitter_ms: float | None = None


# Columns added after the original Phase 2 schema, in the order they were
# introduced. `_init_db` creates them and `_migrate` back-fills any an older
# database is missing, so the two cannot drift apart.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dataset", "TEXT DEFAULT 'synthetic'"),
    ("record_id", "TEXT"),
    ("rpeak_integrity", "REAL"),
    ("qrs_f1", "REAL"),
    ("qrs_sensitivity", "REAL"),
    ("qrs_ppv", "REAL"),
    ("qrs_jitter_ms", "REAL"),
)


class ResultsDB:
    """SQLite backend for recording experiment results."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()
        self._migrate()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT,
                    flicker_freq REAL,
                    input_snr REAL,
                    epochs INTEGER,
                    stdp_tau REAL,
                    seed INTEGER,
                    delta_snr REAL,
                    convergence_speed INTEGER
                )
            """)

    def _migrate(self) -> None:
        """Add missing columns to an existing database.

        Uses ``PRAGMA table_info`` to detect which columns exist and
        ``ALTER TABLE ... ADD COLUMN`` for any in ``_ADDED_COLUMNS`` that are
        missing. New databases are created with the original eight columns and
        brought up to date by this same path, so an old 8-column
        ``results/full_sweep.db`` from Phase 2 and a fresh database end up with
        an identical schema.
        """
        cursor = self.conn.execute("PRAGMA table_info(runs)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        migrations = [
            f"ALTER TABLE runs ADD COLUMN {name} {sql_type}"
            for name, sql_type in _ADDED_COLUMNS
            if name not in existing_columns
        ]

        if migrations:
            with self.conn:
                for sql in migrations:
                    self.conn.execute(sql)

    def insert_result(self, res: RunResult) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs
                (signal_type, flicker_freq, input_snr, epochs,
                 stdp_tau, seed, delta_snr, convergence_speed,
                 dataset, record_id, rpeak_integrity,
                 qrs_f1, qrs_sensitivity, qrs_ppv, qrs_jitter_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    res.signal_type,
                    res.flicker_freq,
                    res.input_snr,
                    res.epochs,
                    res.stdp_tau,
                    res.seed,
                    res.delta_snr,
                    res.convergence_speed,
                    res.dataset,
                    res.record_id,
                    res.rpeak_integrity,
                    res.qrs_f1,
                    res.qrs_sensitivity,
                    res.qrs_ppv,
                    res.qrs_jitter_ms,
                ),
            )


def _reset_transient_state(sim: Simulator) -> None:
    """Reset all transient (non-weight) state for a clean epoch.

    Resets neuron voltages, refractory timers, STDP traces,
    encoder/decoder state, and the simulator clock. Synaptic
    weights are preserved.
    """
    assert isinstance(sim.network, FeedForwardNetwork)
    sim.time = 0.0
    sim.encoder.reset_state()
    sim.decoder.reset_state()
    sim.network.reset_transient_state()


def _compute_convergence_speed(
    epoch_delta_snrs: list[float],
    final_delta_snr: float,
) -> int:
    """Find the first epoch where ΔSNR reaches 90% of final value.

    Args:
        epoch_delta_snrs: Per-epoch ΔSNR measurements.
        final_delta_snr: ΔSNR from the final evaluation pass.

    Returns:
        1-indexed epoch number at which 90% was first reached,
        or total epochs if threshold was never crossed.
    """
    if final_delta_snr <= 0 or not epoch_delta_snrs:
        return len(epoch_delta_snrs) if epoch_delta_snrs else 0

    threshold = 0.9 * final_delta_snr
    for i, dsnr in enumerate(epoch_delta_snrs):
        if dsnr >= threshold:
            return i + 1  # 1-indexed
    return len(epoch_delta_snrs)


def _resolve_signal(params: dict[str, Any], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch signal generation based on the ``dataset`` parameter.

    - ``"synthetic"`` (default): uses :class:`SignalGenerator` — the exact
      Phase 2 path.
    - ``"mit-bih"``: lazily imports :func:`mit_bih.annotator.make_pair`
      so that Phase 2 code never loads ``wfdb``.

    Args:
        params: Task parameter dict (must contain signal-generation keys).
        fs: Sampling frequency in Hz.

    Returns:
        ``(clean, noisy)`` signal pair.
    """
    dataset = params.get("dataset", "synthetic")

    if dataset == "mit-bih":
        from neuroflux_datasets.mit_bih.annotator import make_pair  # noqa: PLC0415

        return make_pair(
            record_id=params["record_id"],
            cache_dir=params["cache_dir"],
            duration_s=params["duration_s"],
            flicker_freq=params["flicker_freq"],
            flicker_amplitude=params.get("flicker_amplitude", 0.5),
            snr_db=params["input_snr"],
            fs=fs,
            seed=params["seed"],
        )

    # Default: synthetic path (byte-identical to Phase 2)
    gen = SignalGenerator(fs=fs, seed=params["seed"])
    return gen.generate(
        signal_type=params["signal_type"],
        duration_s=params["duration_s"],
        flicker_freq=params["flicker_freq"],
        flicker_amplitude=params.get("flicker_amplitude", 0.5),
        snr_db=params["input_snr"],
    )


def _resolve_reference_beats(params: dict[str, Any], fs: float) -> np.ndarray:
    """Cardiologist beat positions for the window ``_resolve_signal`` chose.

    ``make_pair`` selects window ``seed % n_windows``; this applies the same
    rule to the annotated extractor, so the beats and the signal are the same
    window by construction rather than by coincidence.
    """
    from neuroflux_datasets.mit_bih.annotator import (  # noqa: PLC0415
        extract_annotated_segments,
    )

    segments = extract_annotated_segments(
        params["record_id"], params["cache_dir"], params["duration_s"], fs
    )
    if not segments:
        raise ValueError(
            f"Record {params['record_id']} has no artifact-free segment of "
            f"{params['duration_s']}s at {fs} Hz."
        )
    beats: np.ndarray = segments[params["seed"] % len(segments)].beats
    return beats


def run_single_experiment(params: dict[str, Any]) -> RunResult:
    """Executes a single SNN parameter configuration."""
    signal_type = params["signal_type"]
    flicker_freq = params["flicker_freq"]
    input_snr = params["input_snr"]
    epochs = params["epochs"]
    stdp_tau = params["stdp_tau"]
    seed = params["seed"]
    duration_s = params["duration_s"]
    dataset = params.get("dataset", "synthetic")
    record_id = params.get("record_id")

    fs = 250.0
    dt = 1000.0 / fs  # 4.0 ms

    # Network architecture (default preserves Phase 2 / pre-Phase-3 behavior).
    network_layers: list[int] = params.get("network_layers") or [32, 16, 8]
    n_input = network_layers[0]

    # 1. Generate Signal (dispatches on dataset)
    clean, noisy = _resolve_signal(params, fs)

    # Calculate baseline SNR
    base_snr = calculate_snr(clean, noisy)

    # Normalize signal for the SNN
    v_min = float(np.min(noisy))
    v_max = float(np.max(noisy))
    if v_max <= v_min:
        v_max = v_min + 1.0

    # 2. Configure SNN
    net = Network.feedforward(
        layers=network_layers,
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=stdp_tau, tau_minus=stdp_tau),
        seed=seed,
    )

    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=100.0,
        rng=np.random.default_rng(seed),
    )
    decoder = PopulationDecoder(v_min=v_min, v_max=v_max, max_firing_rate_hz=100.0)

    sim = Simulator(net, dt=dt, seed=seed, encoder=encoder, decoder=decoder)

    # SNN expects (T, n_input). Tile single channel across the input layer.
    time_varying_noisy = np.tile(noisy, (n_input, 1)).T

    assert isinstance(sim.network, FeedForwardNetwork)

    # 3. Train SNN with per-epoch convergence tracking
    epoch_delta_snrs: list[float] = []
    duration_ms = duration_s * 1000.0

    for _ in range(epochs):
        # Training pass
        sim.run(time_varying_noisy, duration_ms=duration_ms, training=True)
        _reset_transient_state(sim)

        # Eval pass — save/restore encoder RNG so training is unaffected
        rng_state = encoder.rng.bit_generator.state
        eval_spikes = sim.run(
            time_varying_noisy,
            duration_ms=duration_ms,
            training=False,
        )
        eval_recon = sim.decode(eval_spikes)
        epoch_delta_snrs.append(calculate_snr(clean, eval_recon) - base_snr)
        encoder.rng.bit_generator.state = rng_state
        _reset_transient_state(sim)

    # 4. Final Evaluation Epoch
    out_spikes = sim.run(
        time_varying_noisy,
        duration_ms=duration_ms,
        training=False,
    )
    reconstructed = sim.decode(out_spikes)

    final_snr = calculate_snr(clean, reconstructed)
    delta_snr = final_snr - base_snr

    convergence_speed = _compute_convergence_speed(
        epoch_delta_snrs,
        delta_snr,
    )

    # 5. Clinical landmark scores (MIT-BIH only — synthetic ECG has no
    # annotations, and scoring against the generator's own beat train would
    # measure the generator).
    rpeak_score: float | None = None
    qrs: QrsScore | None = None
    if dataset == "mit-bih":
        from neuroflux_core.metrics.qrs import qrs_score  # noqa: PLC0415
        from neuroflux_core.metrics.rpeak import rpeak_integrity  # noqa: PLC0415

        # Deprecated and saturated; kept so new rows stay comparable with the
        # published Phase 3 columns. See docs/r1_findings.md.
        rpeak_score = rpeak_integrity(clean, reconstructed, fs)
        qrs = qrs_score(_resolve_reference_beats(params, fs), reconstructed, fs)

    return RunResult(
        signal_type=signal_type,
        flicker_freq=flicker_freq,
        input_snr=input_snr,
        epochs=epochs,
        stdp_tau=stdp_tau,
        seed=seed,
        delta_snr=delta_snr,
        convergence_speed=convergence_speed,
        dataset=dataset,
        record_id=record_id,
        rpeak_integrity=rpeak_score,
        qrs_f1=None if qrs is None else qrs.f1,
        qrs_sensitivity=None if qrs is None else qrs.sensitivity,
        qrs_ppv=None if qrs is None else qrs.ppv,
        # SQLite has no NaN; an unmatched window stores NULL instead.
        qrs_jitter_ms=(
            None if qrs is None or math.isnan(qrs.rms_jitter_ms) else qrs.rms_jitter_ms
        ),
    )


def _build_phase2_tasks(config: Phase2Config) -> list[dict[str, Any]]:
    """Build task dicts from a Phase 2 config (Cartesian product)."""
    param_combinations = list(
        itertools.product(
            config.signal_types,
            config.flicker_frequencies,
            config.input_snrs,
            config.training_epochs,
            config.stdp_taus,
            range(config.seeds_per_config),
        )
    )

    tasks: list[dict[str, Any]] = []
    for params in param_combinations:
        tasks.append(
            {
                "signal_type": params[0],
                "flicker_freq": params[1],
                "input_snr": params[2],
                "epochs": params[3],
                "stdp_tau": params[4],
                "seed": params[5],
                "duration_s": config.duration_s,
                "flicker_amplitude": config.flicker_amplitude,
                "network_layers": config.network_layers,
            }
        )
    return tasks


def _build_phase3_tasks(config: Phase3Config) -> list[dict[str, Any]]:
    """Build task dicts from a Phase 3 config (Cartesian product over records)."""
    param_combinations = list(
        itertools.product(
            config.record_ids,
            config.flicker_frequencies,
            config.input_snrs,
            config.training_epochs,
            config.stdp_taus,
            range(config.seeds_per_config),
        )
    )

    tasks: list[dict[str, Any]] = []
    for params in param_combinations:
        tasks.append(
            {
                "dataset": "mit-bih",
                "signal_type": "ecg",  # for analysis.py grouping
                "record_id": params[0],
                "flicker_freq": params[1],
                "input_snr": params[2],
                "epochs": params[3],
                "stdp_tau": params[4],
                "seed": params[5],
                "duration_s": config.duration_s,
                "flicker_amplitude": config.flicker_amplitude,
                "cache_dir": config.cache_dir,
                "network_layers": config.network_layers,
            }
        )
    return tasks


def _build_tasks(raw_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch task building based on the ``dataset`` key in raw YAML."""
    dataset = raw_config.get("dataset", "synthetic")
    if dataset == "mit-bih":
        config = Phase3Config(**raw_config)
        return _build_phase3_tasks(config)
    # Default: Phase 2 synthetic
    config_p2 = Phase2Config(**raw_config)
    return _build_phase2_tasks(config_p2)


def run_experiment_command(args: argparse.Namespace) -> None:
    """Handles the 'run' CLI command."""
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    tasks = _build_tasks(raw_config)

    dataset = raw_config.get("dataset", "synthetic")

    # --- Dry-run mode ---
    if getattr(args, "dry_run", False):
        print(f"Dry run: {len(tasks)} tasks validated.")
        print(f"  Dataset: {dataset}")

        if dataset == "mit-bih":
            config = Phase3Config(**raw_config)
            print(f"  Records: {len(config.record_ids)}")
            print(f"  Flicker frequencies: {config.flicker_frequencies}")
            print(f"  Input SNRs: {config.input_snrs}")
            print(f"  Training epochs: {config.training_epochs}")
            print(f"  STDP taus: {config.stdp_taus}")
            print(f"  Seeds per config: {config.seeds_per_config}")
            print(f"  Duration: {config.duration_s}s")
            print(f"  Cache dir: {config.cache_dir}")

            from neuroflux_datasets.mit_bih.downloader import (  # noqa: PLC0415
                verify_records,
            )

            missing = verify_records(config.record_ids, config.cache_dir)
            if missing:
                print(f"  Missing records ({len(missing)}): {missing}")
            else:
                print("  All records cached.")
        else:
            config_p2 = Phase2Config(**raw_config)
            print(f"  Signal types: {config_p2.signal_types}")
            print(f"  Flicker frequencies: {config_p2.flicker_frequencies}")
            print(f"  Input SNRs: {config_p2.input_snrs}")
            print(f"  Training epochs: {config_p2.training_epochs}")
            print(f"  STDP taus: {config_p2.stdp_taus}")
            print(f"  Seeds per config: {config_p2.seeds_per_config}")
            print(f"  Duration: {config_p2.duration_s}s")

        return

    db = ResultsDB(args.output_db)

    print(f"Starting execution of {len(tasks)} runs...")

    # Execute in parallel
    completed = 0
    failures: list[tuple[dict[str, Any], BaseException]] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_experiment, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                res = future.result()
                db.insert_result(res)
                completed += 1
                if completed % 10 == 0:
                    print(f"Completed {completed} / {len(tasks)} runs...")
            except Exception as e:
                print(f"Error in run {task}: {e!r}")
                failures.append((task, e))

    print(
        f"Finished {completed} / {len(tasks)} runs. Results saved to {args.output_db}"
    )

    if failures:
        first_task, first_exc = failures[0]
        raise RuntimeError(
            f"{len(failures)} / {len(tasks)} runs failed. "
            f"First failure (task={first_task}): {first_exc!r}"
        )


def analyze_significance_command(args: argparse.Namespace) -> None:
    """Handles the 'analyze-significance' CLI command."""
    import csv

    from neuroflux_datasets.analysis import (
        compute_statistical_significance,
        fetch_raw_runs,
        identify_failure_modes,
        summarize_significance,
    )

    db_path = args.db
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_threshold: float = args.p_threshold

    print(f"Loading raw runs from {db_path}...")
    try:
        raw_runs = fetch_raw_runs(db_path)
    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
        return

    if not raw_runs:
        print("No runs found in the database.")
        return

    print(f"Found {len(raw_runs)} raw runs. Computing significance...")
    sig_results = compute_statistical_significance(raw_runs, p_threshold=p_threshold)
    summary = summarize_significance(sig_results)
    failures = identify_failure_modes(sig_results)

    print("\n=== Significance Summary ===")
    print(f"Total configs:        {summary['total_configs']}")
    print(
        f"Significant (p<{p_threshold:.2f}):  "
        f"{summary['significant_configs']} "
        f"({summary['pct_significant']:.1f}%)"
    )
    print(
        f"Improved (ΔSNR>0):    "
        f"{summary['improved_configs']} "
        f"({summary['pct_improved']:.1f}%)"
    )
    print(f"Mean ΔSNR overall:    {summary['mean_delta_snr_overall']:.3f} dB")

    def describe(config: dict[str, Any]) -> str:
        record = config.get("record_id")
        where = f"{config['signal_type']}"
        if record:
            where = f"{config.get('dataset', '?')}/{record}"
        return (
            f"{where}, {config['flicker_freq']} Hz, "
            f"{config['input_snr']} dB SNR, {config['epochs']} ep, "
            f"τ={config['stdp_tau']} → ΔSNR={config['mean_delta_snr']:.3f} dB "
            f"(n={config['n_seeds']})"
        )

    if summary["best_config"]:
        print(f"\nBest:  {describe(summary['best_config'])}")
    if summary["worst_config"]:
        print(f"Worst: {describe(summary['worst_config'])}")

    seed_counts = {r["n_seeds"] for r in sig_results}
    print(f"\nSeeds per config: {sorted(seed_counts)}")
    print(
        "  Unit of analysis is one configuration per record. If this looks like "
        "seeds x records,\n  the grouping key has regressed — see analysis.py "
        "and tests/datasets/test_statistics.py."
    )

    print(f"\nFailure modes: {len(failures)} configs")

    if sig_results:
        sig_path = out_dir / "significance_results.csv"
        with open(sig_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sig_results[0].keys()))
            writer.writeheader()
            writer.writerows(sig_results)
        print(f"Significance results → {sig_path}")

    if failures:
        fail_path = out_dir / "failure_modes.csv"
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        print(f"Failure modes        → {fail_path}")

    print(f"\nAll reports saved to {out_dir}/")


def compare_command(args: argparse.Namespace) -> None:
    """Handles the 'compare' CLI command — the R2 paired protocol."""
    import csv

    from neuroflux_datasets.analysis import fetch_raw_runs, paired_comparison

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_runs = fetch_raw_runs(args.db)
    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
        return
    if not raw_runs:
        print("No runs found in the database.")
        return

    def coerce(text: str) -> Any:
        """Match the column's stored type — argparse hands us strings."""
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
        return text

    rows = paired_comparison(
        raw_runs,
        factor=args.factor,
        level_a=coerce(args.level_a),
        level_b=coerce(args.level_b),
        metric=args.metric,
        stratify_by=args.stratify_by,
        margin=args.margin,
        alpha=args.alpha,
    )
    if not rows:
        print(
            f"No units carry both {args.factor}={args.level_a} and "
            f"{args.factor}={args.level_b}. Nothing to pair."
        )
        return

    def verdict(row: dict[str, Any]) -> str:
        """
        Significance and equivalence are independent, so there are four outcomes.

        Collapsing them loses the one this project needs most: an effect can be
        statistically real *and* smaller than the margin that would make it
        matter. "16x the output population changes the ceiling by 0.16 dB" is
        precisely that case, and calling it either "different" or "equivalent"
        alone would misrepresent it.
        """
        if row["significant"] and not row["equivalent"]:
            return "different"
        if row["significant"] and row["equivalent"]:
            return "trivial"
        if row["equivalent"]:
            return "equivalent"
        return "inconclusive"

    print(
        f"\n=== {args.metric}: {args.factor} {args.level_b} vs {args.level_a} ===\n"
        f"Paired within configuration, seeds averaged. Family size "
        f"{rows[0]['family_size']}, Holm-Bonferroni corrected.\n"
        f"Equivalence margin ±{args.margin} (pre-registered).\n"
    )
    header = (
        f"{'stratum':>10} {'n':>4} {'median Δ':>10} {'95% CI':>18} "
        f"{'p (Holm)':>10} {'verdict':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        interval = (
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            if math.isfinite(row["ci_low"])
            else "n/a"
        )
        stratum = "all" if row["stratum"] is None else f"{row['stratum']}"
        print(
            f"{stratum:>10} {row['n_pairs']:>4} {row['median_diff']:>+10.3f} "
            f"{interval:>18} {row['p_holm']:>10.4f} {verdict(row):>14}"
        )

    print(
        "\n  different    — real effect, larger than the margin\n"
        "  trivial      — real effect, but smaller than the margin\n"
        "  equivalent   — no effect, and bounded inside the margin\n"
        "  inconclusive — neither established. NOT evidence of invariance;\n"
        "                 report it as underpowered."
    )

    out_path = out_dir / (
        f"compare_{args.metric}_{args.factor}_{args.level_a}_vs_{args.level_b}.csv"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


def analyze_experiment_command(args: argparse.Namespace) -> None:
    """Handles the 'analyze' CLI command."""
    from neuroflux_datasets.analysis import (
        fetch_aggregated_data,
        plot_convergence_curves,
        plot_parameter_sensitivity,
        plot_snr_heatmap,
    )

    db_path = args.db
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching and aggregating data from {db_path}...")
    try:
        results = fetch_aggregated_data(db_path)
    except sqlite3.OperationalError as e:
        print(f"Error reading database: {e}")
        return

    if not results:
        print("No results found in the database.")
        return

    signal_types = set(r["signal_type"] for r in results)

    for sig in signal_types:
        print(f"Generating plots for {sig.upper()}...")

        heatmap_path = out_dir / f"snr_heatmap_{sig}.png"
        plot_snr_heatmap(results, sig, heatmap_path)

        convergence_path = out_dir / f"convergence_{sig}.png"
        plot_convergence_curves(results, sig, convergence_path)

        sensitivity_path = out_dir / f"sensitivity_{sig}.png"
        plot_parameter_sensitivity(results, sig, sensitivity_path)

    print(f"All plots saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroFlux Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run experiments from a YAML config."
    )
    run_parser.add_argument("config", type=str, help="Path to the YAML config file.")
    run_parser.add_argument(
        "--output-db",
        type=str,
        default="results.db",
        help="Path to the SQLite output database.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and print task count without running.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze results and generate plots."
    )
    analyze_parser.add_argument("db", type=str, help="Path to the SQLite database.")
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save generated plots.",
    )

    sig_parser = subparsers.add_parser(
        "analyze-significance",
        help="Run statistical significance analysis on results.",
    )
    sig_parser.add_argument("db", type=str, help="Path to the SQLite database.")
    sig_parser.add_argument(
        "--output-dir",
        type=str,
        default="significance",
        help="Directory to save CSV reports.",
    )
    sig_parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="p-value threshold for significance (default: 0.05).",
    )

    cmp_parser = subparsers.add_parser(
        "compare",
        help="Paired comparison of two levels of one factor (R2 protocol).",
    )
    cmp_parser.add_argument("db", type=str, help="Path to the SQLite database.")
    cmp_parser.add_argument(
        "factor", type=str, help="Column to vary, e.g. epochs or stdp_tau."
    )
    cmp_parser.add_argument("level_a", type=str, help="Baseline level.")
    cmp_parser.add_argument("level_b", type=str, help="Comparison level.")
    cmp_parser.add_argument(
        "--metric",
        type=str,
        default="delta_snr",
        help="Column compared (default: delta_snr).",
    )
    cmp_parser.add_argument(
        "--stratify-by",
        type=str,
        default=None,
        help="Config column to stratify on, e.g. input_snr.",
    )
    cmp_parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Equivalence margin in the metric's units (default: 0.5).",
    )
    cmp_parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level."
    )
    cmp_parser.add_argument(
        "--output-dir",
        type=str,
        default="significance",
        help="Directory to save the CSV report.",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_experiment_command(args)
    elif args.command == "analyze":
        analyze_experiment_command(args)
    elif args.command == "analyze-significance":
        analyze_significance_command(args)
    elif args.command == "compare":
        compare_command(args)


if __name__ == "__main__":
    main()
