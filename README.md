# NeuroFlux

NeuroFlux is a Spiking Neural Network (SNN) framework for unsupervised powerline noise suppression in ECG signals. STDP-driven synaptic learning identifies the repeating morphological structure of the QRS complex and passively attenuates interference that does not match it — without labeled data, a reference noise channel, or prior knowledge of the interference frequency.

## Current Status: Phase 2 Complete

**Phase 1 — Core SNN Engine** (`neuroflux_core`) is fully implemented, rigorously tested (97% coverage), and mathematically verified via snapshot hashes.

**Phase 2 — Synthetic Data Validation** is complete. A 5,760-run parameter sweep (576 configs × 10 seeds) across ECG, EEG, and sine signals was executed and analysed:

| Signal | Significant configs (p < 0.01) | Mean ΔSNR |
|--------|-------------------------------|-----------|
| ECG | **192 / 192 — 100%** | **+6.5 dB** |
| EEG | 23 / 192 — 12% | −3.4 dB |
| Sine | 48 / 192 — 25% | −2.0 dB |

**Key finding:** STDP-based suppression works well for morphologically structured signals (ECG). Performance degrades on stochastic multi-band signals (EEG) and purely periodic tones (sine), where the network cannot distinguish signal from flicker. Phase 3 will focus on ECG as the primary modality.

Next: **Phase 3 — Real biomedical dataset integration** (MIT-BIH, PTB-XL).

### Core Engine Features

- **LIF Neurons:** Vectorized leaky integrate-and-fire neurons with membrane voltage, leak decay, and refractory clamping. Core physics runs in a Numba `@njit(cache=True)` kernel.
- **Trace-Based STDP:** O(1) spike-timing-dependent plasticity using exponential pre/post-synaptic traces. Updates weights via LTP and LTD without iterating over spike history.
- **Encoders / Decoders:** `PoissonRateEncoder`, `DeltaEncoder`, and `TemporalEncoder` convert analog signals to spike trains. `PopulationDecoder` and `SpikeRateDecoder` convert output spikes back to analog.
- **Deterministic execution:** Seeded RNG plumbed through `Simulator` and `Network`. Weight checkpointing via `.npz`. SNR, firing rate, and Fano factor metrics included.

## Quickstart

This repository uses [uv](https://github.com/astral-sh/uv) for lightning-fast workspace and dependency management. You must have `uv` installed to use the Make commands.

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and set up the workspace
make install

# Run fast tests only (used in CI — excludes long sweep tests)
make test-fast

# Run the full test suite including slow sweep tests
make test

# Run the Ruff linter and Mypy type checker
make lint
```

### Run the Phase 2 experiment sweep

```bash
# Quick validation (2 runs, seconds)
uv run neuroflux-experiment run configs/phase2/quick_validation.yaml --output-db results/quick.db

# Full sweep (5,760 runs, ~15 min on 8 cores)
uv run neuroflux-experiment run configs/phase2/full_sweep.yaml --output-db results/full_sweep.db

# Statistical analysis + failure mode report
uv run neuroflux-experiment analyze-significance results/full_sweep.db --output-dir results/ --p-threshold 0.01
```

Pre-computed results from the full sweep are in `results/`. Interactive analysis notebook: `experiments/phase2_analysis.ipynb`.

## Repository Structure

- `packages/core/`: SNN engine — LIF neurons, STDP synapses, encoders, decoders, metrics.
- `packages/datasets/`: Phase 2 synthetic signal generators and the `neuroflux-experiment` CLI.
- `packages/baselines/`: Baseline comparison framework — not yet implemented (Phase 4).
- `configs/phase2/`: YAML sweep configs (`full_sweep.yaml`, `quick_validation.yaml`).
- `results/`: Full sweep output — SQLite database, significance CSVs, publication figures.
- `experiments/`: Phase 2 analysis notebook (`phase2_analysis.ipynb`).
- `tests/`: Unit, integration, property-based, and snapshot tests.

For technical specifications see `DETAILS.md`. For the roadmap see `PLAN.md`.
