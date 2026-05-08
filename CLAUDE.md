# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project

**NeuroFlux** — a Python-based Spiking Neural Network (SNN) framework for adaptive flicker suppression in temporal biomedical signals (ECG, EEG, PPG). The mechanism is biologically inspired: STDP-driven unsupervised learning amplifies recurring signal structure and passively suppresses noise, with no labeled training data required.

**Phases 0, 1, and 2 are complete.** Next is Phase 3 (real biomedical dataset integration). See `PLAN.md` for the full 16-week roadmap and `DETAILS.md` for the full technical specification.

---

## Commands

```bash
make install      # uv sync --all-packages
make test         # NUMBA_DISABLE_JIT=1 pytest tests/ packages/ --cov=packages/
make test-fast    # same but -m "not slow" — used in CI, excludes sweep tests
make test-slow    # only @pytest.mark.slow tests (runs actual experiment sweeps)
make lint         # ruff check + mypy (strict)
make format       # ruff format
make clean        # remove __pycache__, .pytest_cache, .ruff_cache, .mypy_cache
```

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
│   ├── core/           # neuroflux-core — Phase 1 (complete)
│   ├── datasets/       # neuroflux-datasets — Phase 2 (complete)
│   └── baselines/      # neuroflux-baselines — Phase 4 (stub only)
├── configs/phase2/     # full_sweep.yaml, quick_validation.yaml
├── results/            # full_sweep.db, significance_results.csv, figures/
├── tests/              # integration + property + snapshot tests
├── experiments/        # phase2_analysis.ipynb
├── docs/               # Architecture docs (empty, Phase 7)
└── pyproject.toml      # uv workspace root; dev deps (pytest, ruff, mypy, mlflow)
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
- `winner_take_all(spike_counts)` → int
- `classification_accuracy(predicted, true)` → float
- `mean_firing_rate(spike_train, dt)` → Hz per neuron
- `silent_neuron_fraction(spike_train)` → float
- `fano_factor(spike_train, dt, window_ms)` → per neuron

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

`Phase2Config` — Pydantic v2 model with fields: `signal_types`, `flicker_frequencies`, `input_snrs`, `training_epochs`, `stdp_taus`, `flicker_amplitude` (default 0.5), `seeds_per_config`, `duration_s`. `run_experiment_command` uses `ProcessPoolExecutor` (scales to `os.cpu_count()`).

`ExperimentResult` — dataclass with `delta_snr` (float) and `convergence_speed` (int).

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

## Critical non-obvious constraints

**`NUMBA_DISABLE_JIT=1` in tests.** The Makefile sets this for `make test`. Without it, Numba JIT compiles on the first call, which adds ~5–10 s of cold-start latency per test session. The performance benchmark test (`test_benchmark.py`) is explicitly skipped when this env var is set — run it separately without the var to validate PLAN.md's "640 synapses, 10k timesteps < 2s" acceptance criterion.

**`g_syn=30.0` synaptic gain is load-bearing.** The LIFConfig `g_syn` parameter scales the weighted dot-product input current to mV-equivalent values. With default weights in [0,1] and typical network sizes, raw dot-product values are far too small to drive neurons past threshold without this gain factor. Changing it without recalibrating will produce silent networks. The regression test `test_all_layers_fire_under_realistic_input` exists specifically to catch this.

**Weight init scheme for STDP: prefer `"scaled"`.** `init_weights()` defaults to `"scaled"` (weights initialized to [w_min, w_min + 0.3*(w_max-w_min)]). `Network.feedforward()` / `FeedForwardNetwork.build()` defaults to `"uniform"`. Small initial weights are recommended for STDP stability — use `init_scheme="scaled"` for new experiments unless there's a specific reason to start from full range.

**Snapshot tests are SHA-256 pinned** (`tests/core/network/test_snapshot.py`). If any change to LIF dynamics, STDP math, or encoder logic shifts the output, the snapshot will break. Update the expected hashes deliberately and document the reason.

**`mypy` runs in strict mode** on the `packages/` tree (excluding `experiments/`, `docs/`, `packages/web/`). All public functions require type annotations. Use `# type: ignore` only at Numba boundaries where `@njit` return types are opaque.

---

## Tooling

- **Package manager:** `uv` (workspace mode). All three packages are editable-installed into the shared `.venv`. Do not use `pip` directly.
- **Linter:** `ruff` (line-length=88, rules E/F/I/UP/B). Format with `ruff format`.
- **Type checker:** `mypy` strict. Types-PyYAML stubs included as dev dep.
- **Testing:** `pytest` + `hypothesis` for property-based tests. Coverage target: ≥85% for `neuroflux-core`.
- **Python version:** 3.11+ (`.python-version` file pins exact version for CI).
- **Experiment tracking:** MLflow is installed; integration not yet wired (Phase 2+).

---

## Commit style

Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`). Do not add `Co-Authored-By` lines.
