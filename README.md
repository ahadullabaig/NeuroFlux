# NeuroFlux

NeuroFlux is a highly optimized, software-based Neuromorphic Framework designed for Adaptive Pattern Recognition and Flicker Suppression in temporal data streams. 

By modeling biological spiking dynamics, NeuroFlux natively processes complex analog signals (like ECG and EEG waves), dynamically learning to suppress flicker noise anomalies via continuous-time synaptic plasticity.

## 🚀 Current Status: Phase 1 Complete

The **Core SNN Engine** (`neuroflux_core`) has been fully implemented, rigorously tested (97% coverage), and mathematically verified via snapshot hashes. We are currently transitioning into **Phase 2 (Python Demonstration)** to integrate physiological datasets and establish benchmarks against traditional ML baselines.

### 🔥 Core Engine Features
The engine is written in pure Python but aggressively optimized to reach C-level execution speeds via LLVM Just-In-Time compilation.

- **Leaky Integrate-and-Fire (LIF) Neurons:** Vectorized arrays modeling precise membrane voltages, continuous leak decays, and hard-bounded refractory clamp periods.
- **Trace-Based STDP Learning:** A computationally elegant Spike-Timing-Dependent Plasticity mechanism that uses exponential scalar traces to mathematically bypass $O(N^2)$ iteration bottlenecks, executing localized Long-Term Potentiation (LTP) and Depression (LTD).
- **Analog-to-Spike I/O Bridge:** Built-in configurable `PoissonRateEncoder`, `DeltaEncoder`, and `TemporalEncoder` map raw analog signals into boolean probabilities, paired with exponential-smoothing `Population` and `SpikeRate` decoders to translate spiking frequencies back into analog structures.
- **Numba LLVM Acceleration:** Core physics integration loops are wrapped in `@njit(cache=True)`, enabling dense multi-layered network simulations to execute tens of thousands of timesteps in a fraction of a second.
- **Reproducible Orchestration:** Top-to-bottom deterministic RNG plumbing via the `Simulator.run()` and `Network.feedforward()` classes, equipped with zero-cost state checkpointing (`.npz`) and analytical metrics (SNR, Spike Statistics, Fano factors).

## 🛠️ Quickstart

This repository uses `uv` for lightning-fast workspace and dependency management.

```bash
# Install dependencies and setup the workspace
make install

# Run the Pytest suite (enforces strict coverage boundaries)
make test

# Run the Ruff linter and Mypy type checker
make lint
```

## 📁 Repository Structure

- `packages/core/`: The core Python SNN engine.
- `packages/datasets/`: Physiological dataset loaders and automated pipelines.
- `packages/baselines/`: PyTorch/Scikit baseline models for benchmarking.
- `packages/web/`: Future React dashboard for browser-side demonstrations.
- `experiments/`: Jupyter notebooks and analytical result logs.
- `tests/`: Property-based hypothesis tests, snapshots, and API validations.

For exact system parameters and biological specifications, see `DETAILS.md`. For the iterative roadmap schedule, see `PLAN.md`.
