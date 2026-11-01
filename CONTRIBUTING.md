# Contributing to NeuroFlux

Bug reports, reproductions and extensions are all welcome. Please open an [issue](https://github.com/ahadullabaig/NeuroFlux/issues) before starting anything large. NeuroFlux is research code behind a published paper, so changes to its numerics or results need extra care (see [Research conventions](#research-conventions)).

## Setup

You need [uv](https://docs.astral.sh/uv/) and `make`.

```bash
git clone https://github.com/ahadullabaig/NeuroFlux.git
cd NeuroFlux
make install          # uv sync --all-packages: neuroflux-core, neuroflux-datasets, dev tools
```

MIT-BIH records are downloaded from PhysioNet the first time you use them and cached in `data/`, which is gitignored. Never commit them. Tests that need a record skip cleanly when it is missing. To cache the record the Phase 3 regression test uses:

```bash
uv run python -c "from neuroflux_datasets.mit_bih.downloader import download_record; download_record('214', 'data/mit_bih')"
```

## Before you push

CI (`.github/workflows/ci.yml`) runs four checks, and all four must pass:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .           # strict, over the whole repository including tests/
make test-fast          # NUMBA_DISABLE_JIT=1, excludes @pytest.mark.slow, 85% coverage gate
```

`make lint` runs only the middle two, so a branch can pass it and still fail CI on formatting. Run `make format` to fix that.

CI runs on pull requests and on pushes to `main`, not on other branches. Branch from `main`, open a pull request, and merge once CI is green. Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.

## Code style

- `ruff` formats and lints, with line length 88 and rule sets E, F, I, UP and B.
- `mypy --strict` applies everywhere except `experiments/` and `docs/`. Every public function needs type annotations. Use `# type: ignore` only at Numba boundaries, where `@njit` return types are opaque.
- Public classes and functions need Google-style docstrings that say what the arguments mean and what is raised.
- Metrics that are not re-exported are imported from their submodule, as in `from neuroflux_core.metrics.snr import calculate_snr`.

## Tests

| Kind | What it guards | Where |
|---|---|---|
| Unit and property | Component correctness and invariants (Hypothesis) | `tests/core/`, `tests/datasets/` |
| Snapshot | Bit-exact network output, pinned by SHA-256 | `tests/core/network/test_snapshot.py` |
| Regression | The canonical Phase 2 and Phase 3 configurations still perform | `tests/test_phase2_regression.py`, `tests/datasets/test_*regression.py` |
| Slow | Real experiment sweeps (`make test-slow`) | marked `@pytest.mark.slow` |
| Benchmark | 640 synapses for 10,000 steps in under 2 s, with JIT on | `tests/core/network/test_benchmark.py` |

Run the benchmark without `NUMBA_DISABLE_JIT`: `uv run pytest tests/core/network/test_benchmark.py -v`.

**If you change the numerics on purpose** (for example `_lif_step`, `_stdp_step` or an encoder), the snapshot test will fail. Check that the new output is physically reasonable, update the pinned hashes, and explain the change in the pull request. Never update the hashes just to make CI pass.

## Research conventions

Each research stage lives in three places with the same name: the code in `scripts/<stage>/`, the outputs in `results/<stage>/`, and the write-up in `docs/<stage>_findings.md`. New stages should follow the same pattern.

- **Decide the analysis before the grid runs.** Write the test, the margin and the decision rule in the grid script's docstring first. Anything decided after seeing the data is labelled **post hoc** wherever it appears.
- **Use the right statistics.** The unit of analysis is the record (n = 16) or, for synthetic signals, the seed. Use paired Wilcoxon signed-rank tests with Holm correction, BCa bootstrap intervals, and two one-sided tests before claiming equivalence. `neuroflux_datasets.analysis` implements all of these.
- **Correct, don't rewrite.** When a later stage overturns an earlier claim, add a dated correction block at the top of the earlier findings document and leave the original text in place.
- **Commit raw grids gzipped** (`*.csv.gz`). The analysis scripts read the compressed copies directly.
- **Say what a number is.** State whether a ceiling is causal or delay-compensated, which weight-init scheme was used, and which QRS detector, with or without cleaning, produced an F1. Do not use `rpeak_integrity` for new results; it is saturated and kept only for Phase 3's published numbers. Score beats with `metrics.qrs.qrs_score`.

## Constraints that are easy to break

- **`g_syn = 30.0`** in `LIFConfig` is load-bearing. Without recalibration, changing it gives silent networks.
- **`init_scheme`**: `Network.feedforward` defaults to `"uniform"` and `init_weights` to `"scaled"`. Results differ between the two, so always say which one you used.
- **Numba kernels** use `@njit(cache=True)`. Clear `__pycache__` after changing one.
- **`paper/`** is not MIT-licensed. `paper/icee2026/` is the accepted IEEE version and must not change, apart from adding the citation and DOI once the paper is published.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
