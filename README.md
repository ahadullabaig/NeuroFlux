# NeuroFlux

NeuroFlux is a Spiking Neural Network (SNN) framework for unsupervised powerline-noise handling in ECG signals. Two complementary mechanisms operate together: the **passive architecture** (LIF membrane filtering + Poisson rate encoding + population decoding) preserves clinical R-peak landmarks at full fidelity across all noise regimes; **STDP-driven synaptic learning** adds amplitude-domain SNR improvement at low input SNR (≤ 0 dB), trading some landmark fidelity for it. No labeled data, no reference noise channel, no prior knowledge of the interference frequency required. Phase 3 (real MIT-BIH ECG) characterized the operating regime precisely — see `docs/phase3_findings.md`.

## Current Status: Phase 3 Complete

**Phase 1 — Core SNN Engine** (`neuroflux_core`) is fully implemented, rigorously tested (97% coverage), and mathematically verified via snapshot hashes.

**Phase 2 — Synthetic Data Validation** is complete. A 5,760-run parameter sweep (576 configs × 10 seeds) across ECG, EEG, and sine signals was executed and analysed:

| Signal | Significant configs (p < 0.01) | Mean ΔSNR |
|--------|-------------------------------|-----------|
| ECG | **192 / 192 — 100%** | **+6.5 dB** |
| EEG | 23 / 192 — 12% | −3.4 dB |
| Sine | 48 / 192 — 25% | −2.0 dB |

**Key finding:** STDP-based suppression works well for morphologically structured signals (ECG). Performance degrades on stochastic multi-band signals (EEG) and purely periodic tones (sine), where the network cannot distinguish signal from flicker. Phase 3 focuses on ECG as the primary modality.

**Phase 3 — Real Biomedical Dataset Integration** is complete with an operating-regime characterization. The framework runs end-to-end on real ECG from the **MIT-BIH Arrhythmia Database**: a `wfdb`-backed downloader with SHA-256 integrity verification (now active on the 16 sweep records), a 360→250 Hz polyphase preprocessor, an annotation-driven clean-segment extractor, and a record-wise train/test splitter. Reconstruction quality is scored both by ΔSNR and by a clinical **R-peak integrity** metric.

A 2,560-run sweep + five diagnostic experiments produced a more nuanced picture than the synthetic-data Phase 2 result:

| Regime | Mean ΔSNR | Mean R-peak integrity |
|---|---|---|
| Input SNR = 0 dB, 10 training epochs | **+0.71 dB** | 0.66 |
| Input SNR = 20 dB, 50 training epochs | −8.20 dB | 0.71 |
| **Passive (untrained) at any input SNR** | −3.6 to −11.4 dB | **1.000** |
| Canonical pinned config (rec 214, 0 dB, 10 ep, τ=10ms) | **+2.26 dB** | **0.985** |

Headline: the SNN architecture has a regime-dependent advantage. The *passive* network already preserves R-peaks perfectly. STDP training adds 3-4 dB of amplitude-domain improvement at every input SNR but damages R-peak fidelity. The canonical positive configuration is pinned in `tests/datasets/test_phase3_regression.py`. **Full analysis: [`docs/phase3_findings.md`](docs/phase3_findings.md).**

Next: **Phase 4 — Baseline comparison framework** (`packages/baselines/`) — classical notch / adaptive filters and a CNN autoencoder benchmarked against trained *and* passive NeuroFlux, stratified by input-SNR class.

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

### Run the Phase 3 experiment (real MIT-BIH ECG)

```bash
# Validate the config and report which records need downloading — no DB, no network
uv run neuroflux-experiment run configs/phase3/quick_validation.yaml --dry-run

# Quick validation on real ECG (downloads + caches MIT-BIH records on first run)
uv run neuroflux-experiment run configs/phase3/quick_validation.yaml --output-db results/phase3_quick.db

# Full MIT-BIH sweep
uv run neuroflux-experiment run configs/phase3/mit_bih_sweep.yaml --output-db results/phase3_sweep.db
```

The same `run` / `analyze-significance` CLI dispatches on a top-level `dataset:` key in the
YAML — `mit-bih` routes through the real-data pipeline; absent or `synthetic` keeps the
unchanged Phase 2 path. Phase 3 analysis notebook: `experiments/phase3_analysis.ipynb`.

## Repository Structure

- `packages/core/`: SNN engine — LIF neurons, STDP synapses, encoders, decoders, metrics (incl. `rpeak_integrity`).
- `packages/datasets/`: Synthetic signal generators (Phase 2), MIT-BIH data layer (`mit_bih/`, Phase 3), and the `neuroflux-experiment` CLI.
- `packages/baselines/`: Baseline comparison framework — not yet implemented (Phase 4).
- `configs/phase2/`, `configs/phase3/`: YAML sweep configs (production + diagnostic variants).
- `scripts/phase3_diagnostics/`: Reproducible diagnostic scripts (manifest population, output-SNR-floor measurement).
- `results/`: Sweep outputs — SQLite databases, significance CSVs, figures. The Phase 3 source-of-truth DB is `phase3_full_sweep.db`; diagnostics live under `phase3_diagnostics/`.
- `experiments/`: Analysis notebooks (`phase2_analysis.ipynb`, `phase3_analysis.ipynb`).
- `tests/`: Unit, integration, property-based, snapshot, and regression tests (Phase 2 and Phase 3).
- `docs/`: `phase3_findings.md` — the canonical Phase 3 results document.

For technical specifications see `DETAILS.md`. For the roadmap see `PLAN.md`. For Phase 3 results see `docs/phase3_findings.md`.
