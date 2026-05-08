# Contributing to NeuroFlux

Thank you for your interest in contributing to NeuroFlux. This document outlines the conventions, processes, and quality standards for all contributions.

## Development Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install all workspace packages in editable mode
git clone https://github.com/UnisysUIP/2026-Neuromorphic-Flicker-Suppression-Using-Software-Simulated-Spiking-Neural-Networks.git
mv 2026-Neuromorphic-Flicker-Suppression-Using-Software-Simulated-Spiking-Neural-Networks/ unisys_project/ && cd unisys_project
make install
```

This runs `uv sync --all-packages`, which installs `neuroflux-core`, `neuroflux-datasets`, and `neuroflux-baselines` as editable packages with all dev dependencies.

## Branch Strategy

We use **trunk-based development** on `main`:

- Create short-lived feature branches from `main` (e.g., `feature/stdp-triplet`, `fix/silent-network-bug`).
- Rebase onto `main` before merging.
- All merges require a passing CI pipeline.

## Pull Request Requirements

Every PR must satisfy the following before merge:

1. **CI Green** — All jobs in `.github/workflows/ci.yml` must pass.
2. **Tests Pass** — `make test` succeeds (runs with `NUMBA_DISABLE_JIT=1`).
3. **Coverage ≥ 85%** — Enforced via `--cov-fail-under=85`.
4. **Lint Clean** — `uv run ruff check .` reports zero errors.
5. **Type Safe** — `uv run mypy .` reports zero errors under strict mode.
6. **Snapshot Awareness** — If your change modifies LIF/STDP numerics, snapshot tests in `tests/core/network/test_snapshot.py` will fail. Update the expected hashes **only** after confirming the change is intentional and documenting why.

## Code Style

### Formatting & Linting

- **Formatter:** `ruff format` (line length = 88).
- **Linter:** `ruff check` with rules `E`, `F`, `I`, `UP`, `B`.
- **Type Checker:** `mypy --strict` on the `packages/` directory.

Run `make lint` to check, `make format` to auto-format.

### Naming Conventions

- **Files:** `snake_case.py`.
- **Classes:** `PascalCase` (e.g., `LIFNeuron`, `STDPSynapse`).
- **Functions/variables:** `snake_case`.
- **Constants:** `UPPER_SNAKE_CASE`.
- **Private helpers:** Prefix with `_` (e.g., `_lif_step`, `_reset_transient_state`).

### Docstrings

All public classes and functions require docstrings. Use Google-style:

```python
def calculate_snr(signal_true: np.ndarray, signal_pred: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in decibels.

    Args:
        signal_true: Clean reference signal.
        signal_pred: Reconstructed or noisy signal.

    Returns:
        SNR value in dB. Returns inf for perfect reconstruction.

    Raises:
        ValueError: If arrays have different shapes.
    """
```

## Testing Guidelines

### Running Tests

```bash
# Standard test run (JIT disabled for speed)
make test

# Performance benchmarks (requires JIT — do NOT set NUMBA_DISABLE_JIT)
uv run pytest tests/core/network/test_benchmark.py -v
```

### Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Unit** | Single component correctness | `test_lif_spike_generation` |
| **Property** | Invariant verification via Hypothesis | `test_lif_voltage_bounds` |
| **Snapshot** | Bit-exact regression on canonical configs | `test_canonical_small_network_snapshot` |
| **Integration** | End-to-end pipeline validation | `test_all_layers_fire_under_realistic_input` |
| **Performance** | Timing benchmarks (JIT required) | `test_640_synapse_performance_benchmark` |

### Snapshot Update Procedure

If you intentionally change the numerics (e.g., modifying `_lif_step` or `_stdp_step`):

1. Run the snapshot test to confirm it fails: `uv run pytest tests/core/network/test_snapshot.py -v`.
2. Inspect the new values and confirm they are physically reasonable.
3. Update the expected hashes and sums in `test_canonical_small_network_snapshot`.
4. Document the reason for the change in your PR description.

## Architecture Overview

```
packages/
├── core/          # SNN engine (neurons, synapses, encoders, decoders, network)
├── datasets/      # Signal generators, experiment runner, analysis
├── baselines/     # Traditional ML/DL comparison models
└── web/           # Future browser dashboard (Phase 5+)
```

### Key Constraints

- **`g_syn = 30.0`** in `LIFConfig` is load-bearing. Do not modify without full recalibration and updating all snapshot tests.
- **`init_scheme="scaled"`** is recommended for STDP stability over `"uniform"`.
- **Numba kernels** (`_lif_step`, `_stdp_step`) use `@njit(cache=True)`. Changes require clearing `__pycache__` to regenerate the cache.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add triplet STDP synapse model
fix: resolve silent-network bug via g_syn scaling
test: add property tests for decoder bounds
chore: update CI to enforce 85% coverage
docs: add architecture diagram to README
```
