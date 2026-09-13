# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project

**NeuroFlux** — a Python-based Spiking Neural Network (SNN) framework for adaptive flicker handling in temporal biomedical signals (ECG, EEG, PPG). Two mechanisms work together: the *passive* architecture (LIF membrane filtering + Poisson rate encoding + population decoding) recovers QRS complexes at low input SNR (≤ 0 dB) that are otherwise undetectable; STDP-driven unsupervised learning adds amplitude-domain SNR improvement at low input SNR. No labeled training data required. The original proposal framed this as a single "STDP denoiser" mechanism — Phase 3's real-data findings showed the passive architecture is doing as much work as STDP, and they make different trade-offs. See `docs/phase3_findings.md` for the precise characterization.

**Phases 0–3 are complete.** Next is Phase 4 (baseline comparison framework — `packages/baselines/`). See `PLAN.md` for the full 16-week roadmap and `DETAILS.md` for the full technical specification.

**Phase 3 produced a regime-characterization result, not the absolute-improvement result the original proposal anticipated.** The full sweep + diagnostic experiments are documented in `docs/phase3_findings.md`. Headline: the SNN has a regime-dependent advantage (improves ΔSNR at input SNR ≤ 0 dB, degrades it at higher SNR); the *passive* untrained network beats the do-nothing baseline on annotation-referenced QRS F1 at ≤ 0 dB and has no effect above 5 dB, and STDP training adds ΔSNR but damages R-peaks. (Phase 3 originally reported "rpeak=1.000 everywhere"; that was a saturated metric — see `docs/r1_findings.md`.) The canonical positive config (record 214, 50 Hz, 0 dB, 10 epochs, τ=10 ms) is pinned in `tests/datasets/test_phase3_regression.py`. Phase 4 plans (in PLAN.md) are revised to compare per input-SNR class and treat passive vs. trained NeuroFlux as an ablation baseline.

**The original "output SNR floor at 10-12 dB" hypothesis was rejected by direct measurement** (`docs/phase3_findings.md` §5.6.1). Running the SNN on a clean signal gives intrinsic output SNR ≈ 0.7 dB, not 10-12. The apparent floor seen in the main sweep is a training-regime effect (noise acts as an implicit regularizer in STDP), not an architectural capacity ceiling. Phase 6 hyperparameter optimization should target the joint (ΔSNR × rpeak) trade-off, **not** "lift the floor" by scaling the output population — `[64, 32, 64]` actually collapses rpeak (see §5.6.3).

---

## Commands

```bash
make install      # uv sync --all-packages
make test         # NUMBA_DISABLE_JIT=1 pytest tests/ packages/ --cov=packages/
make test-fast    # same but -m "not slow" — used in CI, excludes sweep tests
make test-slow    # only @pytest.mark.slow tests (runs actual experiment sweeps)
make lint         # ruff check + mypy . (strict) — see CI warning below
make format       # ruff format
make clean        # remove __pycache__, .pytest_cache, .ruff_cache, .mypy_cache
```

**`make lint` does NOT match CI.** CI runs three gate steps — `ruff format --check .`, `ruff check .`, `mypy .` — but `make lint` only runs the last two. A branch can pass `make lint` and still fail CI on formatting. Before pushing, also run `make format` (or `uv run ruff format --check .`).

Run a single test file:
```bash
NUMBA_DISABLE_JIT=1 uv run pytest tests/core/neurons/test_lif.py -v
```

Run a single test by name:
```bash
NUMBA_DISABLE_JIT=1 uv run pytest -k "test_stdp_causal_spike_pair" -v
```

Run the performance benchmark (requires live Numba JIT — do **not** set `NUMBA_DISABLE_JIT`):
```bash
uv run pytest tests/core/network/test_benchmark.py -v
```

Type-check only the core package:
```bash
uv run mypy packages/core/
```

---

## Repository layout

```
neuroflux/
├── packages/
│   ├── core/                       # neuroflux-core — Phase 1 (complete)
│   ├── datasets/                   # neuroflux-datasets — Phases 2 & 3 (complete)
│   └── baselines/                  # neuroflux-baselines — Phase 4 (stub only)
├── configs/phase2/                 # full_sweep.yaml, quick_validation.yaml
├── configs/phase3/                 # mit_bih_sweep.yaml, quick_validation.yaml,
│                                   # diagnostic_untrained.yaml,
│                                   # diagnostic_network_sweep_{8,16,32,64}.yaml
├── scripts/phase3_diagnostics/     # populate_manifest.py, output_snr_floor.py
├── results/                        # full_sweep.db, significance_results.csv,
│                                   # phase3_full_sweep.db, phase3_full_sig/,
│                                   # phase3_full_figures/, phase3_diagnostics/
├── tests/                          # integration + property + snapshot tests
├── experiments/                    # phase2_analysis.ipynb, phase3_analysis.ipynb
├── docs/                           # phase3_findings.md (Phase 7 fills the rest)
└── pyproject.toml                  # uv workspace root; dev deps
```

`packages/core/src/neuroflux_core/` internal layout follows the PLAN.md module breakdown exactly: `neurons/`, `synapses/`, `encoders/`, `decoders/`, `network/`, `metrics/`, `io/`.

---

## Core engine architecture (`neuroflux_core`)

The public API surface (all re-exported from `neuroflux_core/__init__.py`):

```python
from neuroflux_core import (
    Network, FeedForwardNetwork, Simulator,
    LIFConfig, STDPConfig, NetworkConfig,
    PoissonRateEncoder, TemporalEncoder, DeltaEncoder,
    PopulationDecoder, SpikeRateDecoder,
    save_checkpoint, load_checkpoint,
    init_weights, LIFNeuron, STDPSynapse,
)
```

**Canonical usage:**
```python
net = Network.feedforward(layers=[32, 16, 8], seed=42)
sim = Simulator(net, dt=1.0, seed=42)
output_spikes = sim.run(input_signal, duration_ms=1000.0)
reconstructed = sim.decode(output_spikes)
```

### Data flow

```
analog signal → Encoder → input_spikes (bool[N_in])
                             ↓
              FeedForwardNetwork.step()
                 for each layer:
                   current = weights @ pre_spikes
                   post_spikes = LIFNeuron.step(current, dt)
                   STDPSynapse.step(pre, post, dt)   ← weight update
                 ↓
              output_spikes (bool[N_out]) → Decoder → analog scalar
```

### Key classes

**`LIFNeuron`** (`neurons/lif.py`) — Vectorized group of N LIF neurons. Core physics runs in a Numba `@njit(cache=True)` kernel (`_lif_step`). State: `v` (float32, membrane voltage), `refractory_time` (float32). Default params: V_rest=−70 mV, V_thresh=−55 mV, V_reset=−80 mV, τ_m=20 ms, t_ref=2 ms, g_syn=30.0.

**`STDPSynapse`** (`synapses/stdp.py`) — O(1) trace-based STDP. Core update runs in Numba `@njit(cache=True)` (`_stdp_step`). State: `weights` (float32, shape n_pre×n_post), `trace_pre`, `trace_post`. `reset_state()` restores to `initial_weights_copy` (not zeros), which preserves loaded checkpoints through reset.

**`FeedForwardNetwork`** (`network/feedforward.py`) — For N layers, creates N−1 neuron groups (no input-layer neurons; input is raw spikes) and N−1 synapse matrices. `step()` accepts `training=True/False` to toggle STDP.

**`Simulator`** (`network/simulator.py`) — Wires encoder → network → decoder. When constructed with a `seed`, the Simulator owns a `np.random.Generator` wired into the default `PoissonRateEncoder`. If you pass a custom encoder, you are responsible for seeding it. `run()` accepts 1D (constant per timestep) or 2D (T×N_in) input. `decode()` resets decoder state before iterating — do not call it mid-simulation.

**Configs** (`io/config.py`) — `LIFConfig`, `STDPConfig`, `NetworkConfig` are Pydantic v2 `BaseModel` with `extra="forbid"`. `NetworkConfig.from_yaml()` and `.from_json()` provide file-based configuration.

**Checkpointing** (`io/checkpoint.py`) — Saves/loads weight matrices as `.npz`. Currently only supports `FeedForwardNetwork`. After `load_checkpoint()`, `reset_state()` restores loaded weights (not random initial weights), because `load_checkpoint` also updates `initial_weights_copy`.

### Encoders

| Class | Scheme | Stateful |
|---|---|---|
| `PoissonRateEncoder` | Rate-coded Poisson spikes; probability proportional to normalized amplitude | No |
| `TemporalEncoder` | Time-to-first-spike within a fixed window; high values fire early | Yes (window state) |
| `DeltaEncoder` | Fires on change ≥ threshold from per-channel baseline; modes: absolute/positive/negative | Yes (baseline) |

### Decoders

| Class | Method |
|---|---|
| `PopulationDecoder` | Exponential smoothing trace → mean rate → analog (returns shape `(1,)`) |
| `SpikeRateDecoder` | Sliding-window spike count → mean rate → analog (returns shape `(1,)`) |

### Metrics (`metrics/`)

- `calculate_snr(signal_true, signal_pred)` → dB (primary benchmark metric)
- `rpeak_integrity(clean, reconstructed, fs, tol_ms=50.0)` → float ∈ [0,1] — **deprecated, saturated.** Its prominence floor self-normalises to each signal's std, so it scores the unprocessed noisy signal 1.000 at −5 dB. Kept only because Phase 3's published numbers use it. Do not use for new results.
- `qrs_score(reference_beats, ecg, fs)` → `QrsScore` (Se, PPV, F1, RMS jitter, mean offset) — `metrics/qrs.py`, the replacement. Scores `neurokit2` detections against `.atr` annotations at ±150 ms (EC57). `neurokit2` is lazy-imported and is a dev dependency only. Reference beats come from `neuroflux_datasets.signals.load_reference_beats`.
- `prd`, `prd1`, `rmse`, `max_cross_correlation` — `metrics/distortion.py`. Report `prd1` (baseline-independent), not `prd`.
- `winner_take_all(spike_counts)` → int
- `classification_accuracy(predicted, true)` → float
- `mean_firing_rate(spike_train, dt)` → Hz per neuron
- `silent_neuron_fraction(spike_train)` → float
- `fano_factor(spike_train, dt, window_ms)` → per neuron

**Re-export asymmetry:** `metrics/__init__.py` re-exports **only** `rpeak_integrity` (`__all__ = ["rpeak_integrity"]`). The other five live in submodules and must be imported by full path — e.g. `from neuroflux_core.metrics.snr import calculate_snr`, not `from neuroflux_core.metrics import calculate_snr`. (Inconsistent — a candidate for cleanup, but match the existing pattern until then.)

---

## Phase 2 package (`neuroflux_datasets`)

### Synthetic signal generators (`synthetic.py`)

```python
from neuroflux_datasets.synthetic import (
    generate_ecg, generate_eeg, generate_sine,
    add_flicker, add_gaussian_noise,
    SignalGenerator,
)
```

`SignalGenerator(fs, seed)` is the unified pipeline: call `.generate(signal_type, duration_s, flicker_freq=None, flicker_amplitude=0.0, snr_db=None, **kwargs)` → returns `(clean, noisy)` as `np.ndarray` pairs.

### Experiment runner (`cli.py`)

```bash
uv run neuroflux-experiment run <config.yaml> --output-db <path>
uv run neuroflux-experiment analyze-significance <db> --output-dir <dir> --p-threshold 0.01
```

`Phase2Config` — Pydantic v2 model with `extra="forbid"` and fields: `signal_types`, `flicker_frequencies`, `input_snrs`, `training_epochs`, `stdp_taus`, `flicker_amplitude` (default 0.5), `seeds_per_config`, `duration_s`, `network_layers` (default `[32, 16, 8]`). `run_experiment_command` uses `ProcessPoolExecutor` (scales to `os.cpu_count()`) and **raises `RuntimeError` if any per-task run failed** (loud failures — added post-Phase-3 for Phase 4 readiness). Successful rows are still inserted before the raise.

`RunResult` (`cli.py`) — the result dataclass for **both** Phase 2 and Phase 3 runs. Fields: `signal_type`, `flicker_freq`, `input_snr`, `epochs`, `stdp_tau`, `seed`, `delta_snr`, `convergence_speed`, plus the Phase 3 additions `dataset` (default `"synthetic"`), `record_id` (default `None`), `rpeak_integrity` (default `None`). The defaults are what keep Phase 2 callers valid — see the Phase 3 section. `insert_result` is a fixed-column INSERT, so it must change in lockstep with this dataclass.

### Analysis (`analysis.py`)

| Function | Purpose |
|---|---|
| `fetch_aggregated_data(db)` | AVG delta_snr over seeds, grouped by config |
| `fetch_raw_runs(db)` | All individual rows without aggregation |
| `compute_statistical_significance(raw_runs, p_threshold)` | One-sample t-test per config, Cohen's d |
| `summarize_significance(sig_results)` | High-level totals, best/worst config |
| `identify_failure_modes(sig_results)` | Configs with negative_improvement / not_significant / high_variance |
| `plot_snr_heatmap` / `plot_convergence_curves` / `plot_parameter_sensitivity` | Save PNG figures |

### Phase 2 known limitations

**EEG and sine signals are degraded, not improved.** The full sweep showed ECG at 100% significant configs (mean ΔSNR +6.5 dB) but EEG at 12% and sine at 25%, both with negative mean ΔSNR. STDP learns repeating morphological structure (QRS complex) but cannot distinguish periodic flicker from a periodic signal or latent EEG rhythms. This is a documented finding, not a bug — do not attempt to "fix" it by tuning parameters without a clear mechanistic rationale.

**CI regression test** (`tests/datasets/test_regression.py`) guards the canonical ECG config: `signal_type=ecg`, `flicker_freq=50 Hz`, `input_snr=10 dB`, `epochs=100`, `stdp_tau=20 ms`, `seed=0` must achieve ΔSNR ≥ 4 dB. This runs in `make test-fast` (not slow-marked). If it breaks, a change to LIF dynamics, STDP, encoders, or signal generation has regressed ECG performance.

---

## Phase 3 package (`neuroflux_datasets.mit_bih`)

Real biomedical dataset integration — MIT-BIH Arrhythmia Database (ECG). Modules: `downloader.py` (fetch + cache + SHA-256 verify), `preprocessor.py` (resample 360→250 Hz, normalize), `annotator.py` (`extract_clean_segments`, `make_pair`), `loader.py` (`train_test_split`, record-wise). The experiment runner dispatches on a top-level `dataset:` key in the YAML config — `mit-bih` → `Phase3Config` → `_build_phase3_tasks`; absent/`synthetic` → the unchanged Phase 2 path. `wfdb` is **lazy-imported** everywhere — Phase 2 code never loads it. Downloaded records cache to a gitignored `data/` dir; never commit them.

`Phase3Config` — Pydantic v2 model with `extra="forbid"`: `dataset` (`"mit-bih"`), `record_ids`, `flicker_frequencies`, `input_snrs`, `training_epochs`, `stdp_taus`, `flicker_amplitude` (default 0.5), `seeds_per_config`, `duration_s` (default 10.0), `cache_dir` (default `"data/mit_bih"`), `network_layers` (default `[32, 16, 8]`). Phase 3 task dicts also inject `signal_type="ecg"` so `analysis.py`'s `signal_type` grouping still works. `run` gained a `--dry-run` flag (validates the config, prints task count, calls `verify_records` — opens no DB, runs no simulation).

**`network_layers` field** — shared by both Phase2Config and Phase3Config. Default `[32, 16, 8]` preserves all pre-Phase-3 behavior (Phase 2 regression test verified byte-identically). `run_single_experiment` uses `params["network_layers"][0]` to tile the single-channel input across the input layer, so changing this field correctly resizes the input tile. Phase 3 §5.6.3 found `[32, 32, 32]` outperforms the default on real ECG; this is the recommended starting point for Phase 6 hyperparameter optimization. Wide configurations (input layer ≥ 64) collapse rpeak — see `docs/phase3_findings.md` §5.6.3.

**`ResultsDB` schema migration.** `RunResult` gained `dataset`/`record_id`/`rpeak_integrity` (all with defaults). `ResultsDB._migrate()` runs `PRAGMA table_info(runs)` + `ALTER TABLE ... ADD COLUMN` on open, upgrading an existing pre-Phase-3 8-column `results/full_sweep.db` in place — old rows stay readable, the migration is idempotent. `rpeak_integrity` is computed **only** on the `mit-bih` branch; synthetic runs leave it `None`.

### Phase 3 known limitations / notes for later phases

**SHA-256 integrity verification is now partially active.** 16 of 48 manifest entries carry real checksums — specifically the records used in the Phase 3 sweep (100, 101, 103, 105, 108, 112, 113, 119, 200, 207, 212, 214, 228, 231, 232, 234). The remaining 32 records keep `"dat_checksum": null` and continue to trust the cache as before. To extend coverage: download additional records, then run `uv run python scripts/phase3_diagnostics/populate_manifest.py` (idempotent — re-runs lift any new provenance checksums into the manifest, refuse to overwrite mismatches).

**Phase 3 produced a regime-characterization result.** Full analysis: `docs/phase3_findings.md`. Key facts every contributor should know:
- The SNN does **not** behave as a broad denoiser on real ECG. Mean ΔSNR is negatively correlated with input SNR (improves at ≤ 0 dB, degrades at ≥ 5 dB).
- The passive (untrained, epochs=0) network's `rpeak_integrity = 1.000` in `docs/phase3_findings.md` §5.6.2 is a **metric artifact** — the identity baseline scores 1.000 too. Re-measured with `metrics/qrs.py` against `.atr` annotations, passive beats do-nothing on QRS F1 at −5 dB (+0.37) and 0 dB (+0.22), with no effect at ≥ 5 dB. STDP training adds ~3-4 dB of ΔSNR but damages R-peaks. M1b's gain calibration collapses QRS F1 to ≈0. See `docs/r1_findings.md`.
- Phase 4 should treat *passive NeuroFlux* (configs with `training_epochs: [0]`) as its own baseline alongside the classical filters.
- Phase 4 comparisons must stratify by input SNR class (0, 5, 10, 20 dB) — the regime advantage is not visible in global aggregates.

**Phase 3 regression test** (`tests/datasets/test_phase3_regression.py`) pins the canonical positive config: record 214, 50 Hz flicker, 0 dB input SNR, 10 epochs, τ=10 ms, seed=0, 10 s duration. Asserts `delta_snr ≥ 1.5 dB` and `rpeak_integrity ≥ 0.85`. **Skips cleanly if `data/mit_bih/214.dat` is absent** (so CI without bundled data still passes); runs locally if the record is cached. Pre-cache with `uv run python -c "from neuroflux_datasets.mit_bih.downloader import download_record; download_record('214', 'data/mit_bih')"`.

**Diagnostic scripts and configs** live at `scripts/phase3_diagnostics/` and `configs/phase3/diagnostic_*.yaml`. They are reproducible (deterministic seeds, results checked into `results/phase3_diagnostics/`). If you need to re-establish the Phase 3 mechanistic claims, re-run them in this order: `populate_manifest.py` → `output_snr_floor.py` → `diagnostic_untrained.yaml` → `diagnostic_network_sweep_{8,16,32,64}.yaml`. Total wall time ≈ 10 min on 16 cores.

**`extract_clean_segments` resamples each window individually**, not the whole record once — this introduces minor polyphase edge artifacts at window boundaries. Negligible for the 5–10 s windows used in Phase 3 configs; revisit if shorter windows or boundary-sensitive metrics are added.

**Annotation symbol `+` (rhythm change) is treated as an artifact** in `_ARTIFACT_SYMBOLS` alongside `~` and `|`. In MIT-BIH `+` marks rhythm changes, not noise — this is a deliberately conservative choice that discards some usable clean windows. Defensible, but if clean-segment yield is too low for a record, relax this set.

**`run_experiment_command` swallows per-run exceptions** (`except Exception` → prints `Error in run`). A single bad run (e.g. an R-peak length mismatch from a non-integer `duration_s`, or a missing cached record) is logged and skipped, not raised. Keep `duration_s` integer-valued and pre-cache records (`--dry-run` reports missing ones) so silent drops don't go unnoticed.

---

## Critical non-obvious constraints

**`NUMBA_DISABLE_JIT=1` in tests.** The Makefile sets this for `make test`. Without it, Numba JIT compiles on the first call, which adds ~5–10 s of cold-start latency per test session. The performance benchmark test (`test_benchmark.py`) is explicitly skipped when this env var is set — run it separately without the var to validate PLAN.md's "640 synapses, 10k timesteps < 2s" acceptance criterion.

**`g_syn=30.0` synaptic gain is load-bearing.** The LIFConfig `g_syn` parameter scales the weighted dot-product input current to mV-equivalent values. With default weights in [0,1] and typical network sizes, raw dot-product values are far too small to drive neurons past threshold without this gain factor. Changing it without recalibrating will produce silent networks. The regression test `test_all_layers_fire_under_realistic_input` exists specifically to catch this.

**Weight init scheme for STDP: prefer `"scaled"`.** `init_weights()` defaults to `"scaled"` (weights initialized to [w_min, w_min + 0.3*(w_max-w_min)]). `Network.feedforward()` / `FeedForwardNetwork.build()` defaults to `"uniform"`. Small initial weights are recommended for STDP stability — use `init_scheme="scaled"` for new experiments unless there's a specific reason to start from full range.

**Snapshot tests are SHA-256 pinned** (`tests/core/network/test_snapshot.py`). If any change to LIF dynamics, STDP math, or encoder logic shifts the output, the snapshot will break. Update the expected hashes deliberately and document the reason.

**`mypy` runs in strict mode over the whole repo** — CI runs `uv run mypy .`, which checks `packages/` **and `tests/`** and everything else, excluding only `experiments/`, `docs/`, `packages/web/` (the `[tool.mypy]` exclude list). Test files are type-checked too, so a `**dict[str, object]` splat or a stale `# type: ignore` in `tests/` is a CI failure. The `uv run mypy packages/core/` command above is a fast convenience check **only** — it is not what CI enforces; always run `mypy .` before pushing. All public functions require type annotations. Use `# type: ignore` only at Numba boundaries where `@njit` return types are opaque.

---

## Tooling

- **Package manager:** `uv` (workspace mode). All three packages are editable-installed into the shared `.venv`. Do not use `pip` directly.
- **Linter:** `ruff` (line-length=88, rules E/F/I/UP/B). Format with `ruff format`.
- **Type checker:** `mypy` strict. Types-PyYAML stubs included as dev dep.
- **Testing:** `pytest` + `hypothesis` for property-based tests. Coverage gate: `--cov=packages/ --cov-fail-under=85` — **85% across all of `packages/`**, not just core (enforced by `make test`/`make test-fast`).
- **Python version:** 3.11+ (`.python-version` file pins exact version for CI).
- **Experiment tracking:** MLflow is installed; integration not yet wired (Phase 2+).

---

## Continuous integration

CI (`.github/workflows/ci.yml`) runs four steps, in order: `ruff format --check .` → `ruff check .` → `mypy .` → `make test-fast`. All four must pass. `make lint` covers only the middle two (see the warning under Commands).

**CI only triggers on pull requests and pushes to `main`.** Feature branches (`phase-N`, etc.) never run CI on their own — a branch can sit broken indefinitely and you won't know until you open the PR. Before opening a PR, run all four gate steps locally: `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy .`, `make test-fast`.

---

## Commit style

Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`). Do not add `Co-Authored-By` lines.
