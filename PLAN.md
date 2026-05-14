# NeuroFlux: Implementation Plan

**Project:** Software-based neuromorphic framework for adaptive pattern recognition and flicker suppression
**Plan version:** 1.0
**Target delivery:** 16 weeks (full scope), with staged value delivery every 2–3 weeks

---

## Guiding principles

Before diving into phases, three principles shape this plan:

1. **Working software at every checkpoint.** Each phase ends with a demonstrable, runnable artifact — not a deliverable pending integration. If Phase 3 slips, Phase 2's output is still usable and valuable on its own.

2. **Validate early, scale later.** The interactive browser demo already exists as proof of concept. The next move is not to scale it — it is to validate the science on real biomedical data. Only after that do we optimize, productionize, and extend.

3. **Contract-first design.** Every phase defines its interface (data schemas, API signatures, output formats) before implementation begins. This lets phases be developed in parallel where possible and prevents integration cliff edges.

---

## Phase overview

| Phase | Name | Duration | Deliverable | Dependencies |
|---|---|---|---|---|
| 0 | Foundations and repo setup | 1 week | Monorepo, CI/CD, coding standards | — |
| 1 | Core SNN engine (Python) | 2 weeks | Production-grade `neuroflux-core` package | Phase 0 |
| 2 | Synthetic data validation | 1.5 weeks | Reproducible SNR benchmarks on synthetic signals | Phase 1 |
| 3 | Real biomedical dataset integration | 2 weeks | Pipelines for PhysioNet, TUH-EEG, MIT-BIH | Phase 1 |
| 4 | Baseline comparison framework | 1.5 weeks | Head-to-head results vs. Butterworth, LMS, wavelet, CNN | Phase 2, 3 |
| 5 | Enhanced interactive dashboard | 2 weeks | Full-featured web UI, real-data upload, export | Phase 1, 3 |
| 6 | Hyperparameter optimization | 1 week | Auto-tuned models per signal type | Phase 4 |
| 7 | Documentation, paper draft, handoff | 2 weeks | Research paper, technical docs, onboarding guide | All prior |
| 8 *(optional)* | Neuromorphic hardware port | 3 weeks | Loihi 2 / Xylo deployment | Phase 4 |

**Total core scope:** 13 weeks. **With optional Phase 8:** 16 weeks.

---

## Phase 0 — Foundations and repository setup

**Duration:** 1 week
**Goal:** Establish the engineering substrate so every subsequent phase is fast and low-friction.

### Objectives

Set up the monorepo, tooling, CI/CD, and engineering conventions that will carry the project from prototype to publication.

### Deliverables

A monorepo with the following structure:

```
neuroflux/
├── packages/
│   ├── core/               # Python SNN engine (Phase 1)
│   ├── datasets/           # Dataset loaders (Phase 3)
│   ├── baselines/          # Comparison filters (Phase 4)
│   └── web/                # React dashboard (Phase 5)
├── experiments/            # Jupyter notebooks, result logs
├── docs/                   # Architecture, API reference
├── tests/                  # Integration tests across packages
├── .github/workflows/      # CI pipelines
└── pyproject.toml
```

### Technical decisions to finalize

- **Python version:** 3.11+ (required for performance improvements in numerical code)
- **Core dependencies:** NumPy, SciPy, Numba (JIT compilation for spike loops), PyTorch (for baseline CNN), matplotlib for offline plots
- **Testing:** pytest + hypothesis for property-based testing of neuron dynamics
- **Linting:** ruff (fast), mypy (type checking in strict mode on core package)
- **Documentation:** Sphinx + MyST Markdown
- **Experiment tracking:** Weights & Biases or MLflow (both offline-capable)
- **Data versioning:** DVC for large biomedical datasets

### Acceptance criteria

- `make test` runs all tests in under 60 seconds.
- CI pipeline green on every PR — runs linting, type-checking, unit tests, and coverage.
- One-command environment setup: `uv sync` or `poetry install` produces a working dev environment.
- Repository has a README with clear project vision and phase roadmap.
- License decision made (recommend MIT for maximum academic and commercial applicability).

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-engineering tooling before any science is done | Time-box to 1 week hard. If a tool is not pulling its weight, drop it. |
| Dependency conflicts with biomedical libraries (MNE, wfdb) | Pin versions up front; maintain a lockfile. |

---

## Phase 1 — Core SNN engine (Python)

**Duration:** 2 weeks
**Goal:** Rewrite the in-browser prototype as a production-grade, vectorized Python library with clean APIs, comprehensive tests, and performance suitable for real datasets.

### Why Python, when we already have JavaScript?

The browser demo is for exploration and communication. Research validation, baseline comparison, and batch experiments need the Python scientific stack — NumPy for vectorization, SciPy for signal processing baselines, PyTorch for deep learning baselines, and the entire biomedical ecosystem (MNE-Python, wfdb, neurokit2). The JavaScript version is preserved as the demo; Python is the workhorse.

### Module breakdown

```
neuroflux-core/
├── neurons/
│   ├── lif.py              # Vectorized LIF (handles N neurons simultaneously)
│   ├── izhikevich.py       # Optional richer dynamics for Phase 6+
│   └── base.py             # Abstract Neuron interface
├── synapses/
│   ├── stdp.py             # Pair-based STDP with trace formulation
│   ├── triplet_stdp.py     # Pfister-Gerstner triplet rule (optional)
│   └── base.py
├── encoders/
│   ├── rate.py             # Poisson rate encoding
│   ├── temporal.py         # Time-to-first-spike encoding
│   └── delta.py            # Step-forward encoding for continuous signals
├── decoders/
│   ├── population.py       # Multi-neuron averaging
│   └── spike_rate.py       # Windowed rate decoding
├── network/
│   ├── feedforward.py      # 3-layer reference architecture
│   ├── topology.py         # Configurable connection patterns
│   └── simulator.py        # Main simulation loop
├── metrics/
│   ├── snr.py
│   ├── accuracy.py
│   └── spike_stats.py
└── io/
    ├── config.py           # YAML/JSON config loading
    └── checkpoint.py       # Save/load trained weights
```

### Critical engineering requirements

**Vectorization.** The browser prototype uses per-neuron loops. Python must vectorize all neuron and synapse updates over batched timesteps. A network of 640 synapses simulating 10,000 timesteps should run in under 2 seconds on a laptop CPU.

**Numerical stability.** Explicit Euler integration can become unstable at large dt. Implement validation that flags when membrane voltage exceeds physiological bounds and suggest smaller dt.

**Determinism.** Every simulation accepts a random seed. Running the same config twice produces bit-identical outputs. This is non-negotiable for scientific reproducibility.

**Configuration as data.** Network topology, neuron parameters, and STDP rules are specified in YAML config files, not code. This lets non-programmers run experiments and lets experiments be version-controlled cleanly.

### API contract (finalize in week 1)

```python
from neuroflux import Simulator, Network, LIFConfig, STDPConfig

net = Network.feedforward(
    layers=[32, 16, 8],
    neuron=LIFConfig(tau_m=20, v_thresh=-55),
    plasticity=STDPConfig(a_plus=0.01, a_minus=0.012, tau=20),
)

sim = Simulator(net, dt=1.0, seed=42)
output_spikes = sim.run(input_signal, duration_ms=1000)
reconstructed = sim.decode(output_spikes, method="population")
```

### Test strategy

- **Unit tests:** Each neuron model, each synapse model, each encoder — minimum 90% line coverage.
- **Property tests:** LIF neuron must never fire without input current exceeding threshold. STDP weights must stay within [w_min, w_max]. Membrane voltage must decay to V_rest in absence of input (with tolerance).
- **Integration tests:** Three-layer network on known synthetic inputs produces expected spike counts within statistical bounds.
- **Regression tests:** Outputs of canonical configs are snapshot-tested. Any change must be deliberate and documented.

### Acceptance criteria

- All modules documented with type hints and docstrings.
- Performance target: 640-synapse network, 10,000 timesteps, under 2 seconds on M-series or i7-class CPU.
- Feature parity with the browser demo (produces equivalent SNR improvement on the same synthetic inputs).
- Published as an installable package: `pip install -e packages/core` works cleanly.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Vectorization bugs where per-neuron logic silently desynchronizes | Cross-validate against the JavaScript reference on 10 canonical test cases |
| Numba JIT compilation overhead on first run | Pre-warm in benchmarks; document expected first-call latency |
| API churn during development blocking downstream phases | Freeze public API at end of week 1; only internals change after that |

---

## Phase 2 — Synthetic data validation

**Duration:** 1.5 weeks
**Goal:** Rigorously characterize NeuroFlux's flicker suppression performance on synthetic signals with known ground truth, across a systematic sweep of noise conditions.

### Why this phase exists

Real data validation (Phase 3) is expensive and noisy. Before spending that budget, we need to know: does NeuroFlux work at all on signals where we control every variable? If performance is poor on synthetic data, no amount of real data will fix it.

### Experiment design

Three signal classes:

1. **ECG-like:** QRS complex morphology with configurable heart rate, P-wave amplitude, and baseline wander.
2. **EEG-like:** Multi-band sinusoidal superposition (alpha + beta + gamma) with optional epileptiform spike injection.
3. **Pure sinusoidal:** Single-frequency reference signal for mechanism isolation.

Three noise classes, applied in all combinations:

1. **Flicker (periodic):** Sinusoidal interference at 50 Hz, 60 Hz, 100 Hz, 120 Hz.
2. **Gaussian white noise:** SNR levels of 0 dB, 5 dB, 10 dB, 20 dB.
3. **Mixed:** Flicker + Gaussian at multiple intensities.

### Parameter sweep matrix

| Variable | Values | Count |
|---|---|---|
| Signal type | ECG, EEG, sine | 3 |
| Flicker frequency | 50, 60, 100, 120 Hz | 4 |
| Input SNR | 0, 5, 10, 20 dB | 4 |
| Training epochs | 10, 50, 100, 500 | 4 |
| STDP τ | 10, 20, 40 ms | 3 |
| **Total configurations** | | **576** |

Each configuration runs with 10 random seeds → **5,760 total runs**. At 2 seconds per run on a single core, this is ~3.2 hours sequential, or ~15 minutes on an 8-core machine with parallelization.

### Deliverables

- Automated experiment runner: `neuroflux-experiment run configs/phase2/*.yaml`
- Results database (SQLite or Parquet) with all metrics per run.
- Analysis notebook producing publication-quality figures:
  - SNR improvement heatmap (input SNR × flicker frequency)
  - Convergence curves (SNR vs. training epochs)
  - Parameter sensitivity analysis (which hyperparameters matter most)
- Automated regression test: the canonical config must achieve ΔSNR ≥ 4 dB on the reference synthetic ECG — this runs in CI on every commit.

### Acceptance criteria

- Reproducible: any team member can rerun any experiment from config and get identical numbers.
- Results show statistically significant SNR improvement (p < 0.01, paired t-test) on at least 80% of configurations.
- Failure modes identified: for which configurations does NeuroFlux fail, and why? A "known limitations" section is as important as success claims.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic signals too idealized to predict real-world performance | Deliberately make them harder than reality — lower SNR, more diverse flicker patterns |
| Cherry-picking successful configurations | Pre-register the evaluation protocol; report all configurations, not just winners |
| Statistical significance from large N masking small effect sizes | Report effect sizes (Cohen's d) alongside p-values |

---

## Phase 3 — Real biomedical dataset integration

**Duration:** 2 weeks
**Goal:** Validate NeuroFlux on real, clinically relevant datasets with actual pathological noise characteristics.

### Target datasets

| Dataset | Modality | Size | License | Flicker relevance |
|---|---|---|---|---|
| MIT-BIH Arrhythmia Database | ECG | 48 records × 30 min | Open (PhysioNet) | Powerline noise well-documented |
| PTB-XL | ECG | 21,837 records | Open (PhysioNet) | Large-scale, diverse noise profiles |
| TUH EEG Corpus | EEG | 30,000+ sessions | Research use | Muscle, powerline, electrode noise |
| CHB-MIT | EEG (pediatric) | 23 subjects | Open | Epilepsy monitoring with artifacts |
| Sleep-EDF | EEG | 197 recordings | Open | Long recordings reveal adaptation |

**Recommendation:** Start with MIT-BIH (well-characterized, small, fast iteration) before scaling to PTB-XL.

### Data pipeline architecture

```
[Raw dataset] → [Downloader/cacher] → [Preprocessor] → [Annotator] → [Trainer] → [Evaluator]
       ↓               ↓                    ↓              ↓             ↓           ↓
     DVC           SHA-256 check       Resampling      Clean regions   SNN train   Metrics
                                       Normalization   Noise regions   + STDP      + reports
```

### Preprocessing requirements

- Resample to a consistent sampling rate (250 Hz for ECG, 256 Hz for EEG).
- Normalize to zero mean, unit variance per channel.
- Extract "clean" and "noisy" segments using existing annotations where available; otherwise use SNR-based heuristic labeling.
- Split into train/val/test by **subject** (never by time-window within subject — this leaks information).

### Ground truth challenge

Real signals do not come with clean references. Two approaches, used together:

1. **Artificial corruption of clean segments.** Take high-quality clean segments, add known synthetic flicker, and measure recovery. This gives perfect ground truth but may not capture real noise characteristics.

2. **Clinician annotation.** For a representative subset (20–50 recordings), have a domain expert annotate regions they judge as clean vs. noisy. Use these annotations to measure artifact rejection qualitatively. Budget 10 hours of clinician time total.

### Deliverables

- Dataset package: `neuroflux-datasets` with loaders for all target datasets.
- Preprocessing pipelines with full provenance tracking.
- Validation report: NeuroFlux performance on each dataset, compared to noisy baseline.
- Subject-held-out evaluation protocol documented.

### Acceptance criteria

- At least one real dataset integrated end-to-end.
- Reproducible results from raw dataset download to final metric (one-command pipeline).
- Performance on at least one dataset shows meaningful improvement over noisy input (ΔSNR ≥ 3 dB **or** qualitative artifact reduction confirmed by clinician **or** R-peak preservation ≥ 0.85 in at least one input-SNR regime).

### Outcome (post-Phase-3-sweep update)

Phase 3 completed with a regime-characterization result rather than the originally-anticipated absolute-improvement result. See `docs/phase3_findings.md` for full analysis. Summary:

- **Acceptance criterion met conditionally:** record 214 with the canonical config (50 Hz flicker, 0 dB input SNR, 10 epochs, τ=10 ms) achieves ΔSNR ≈ +2.26 dB and R-peak integrity ≈ 0.985. Pinned in `tests/datasets/test_phase3_regression.py`.
- **Operating regime:** the SNN improves ΔSNR at low input SNR (≤ 0 dB) and degrades it at high input SNR (≥ 5 dB). The passive (untrained) configuration preserves R-peaks perfectly at every input SNR; STDP training adds 3-4 dB of ΔSNR uniformly but at the cost of R-peak fidelity (0.51-0.89).
- **Mechanism revision:** the initial "output SNR floor at 10-12 dB set by population decoder capacity" hypothesis was **rejected** by the §5.6.1 diagnostic — direct measurement on a clean signal gives intrinsic output SNR ≈ 0.7 dB. The actual mechanism is *training-noise-as-regularizer*: STDP trained on noisy input produces a less specialized network than STDP trained on clean input, which is why the apparent "floor" tracks input SNR. This rules out the original "scale the output layer to lift the floor" Phase 6 plan; see §6.2 below for the revised approach.
- **Phase 4-7 plans below are revised** to reflect this finding.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Real noise characteristics differ substantially from synthetic | Iterate — use insights to refine Phase 2 synthetic benchmarks |
| Clinician time unavailable | Fall back to artificial corruption method; note this limitation in results |
| Dataset licensing restricting commercial use | Focus on permissively-licensed datasets (MIT-BIH, PTB-XL, Sleep-EDF) |
| Privacy/HIPAA concerns | Use only de-identified public datasets; document compliance |

---

## Phase 4 — Baseline comparison framework

**Duration:** 1.5 weeks
**Goal:** Compare NeuroFlux head-to-head against the methods clinicians and engineers actually use today. Without this, no reviewer will take the claims seriously.

### Baselines to implement

| Baseline | Why it matters | Implementation |
|---|---|---|
| 2nd-order Butterworth notch filter | Clinical default for powerline | scipy.signal |
| Higher-order Butterworth bandpass | Common for EEG/ECG | scipy.signal |
| LMS adaptive filter | Current best for adaptive removal | Custom (well-documented algorithm) |
| Stationary wavelet transform + soft thresholding | Popular in ECG denoising literature | pywavelets |
| 1D CNN autoencoder | Deep learning baseline | PyTorch, ~10k params |
| Identity (no filtering) | Sanity check / lower bound | Trivial |
| **Passive NeuroFlux (epochs=0)** — *added post-Phase-3* | Ablation: what does STDP add over the SNN architecture alone? | Reuses existing CLI with `training_epochs: [0]`. See `docs/phase3_findings.md` §5.6.2. |

### Evaluation protocol (revised post-Phase-3)

Every baseline is evaluated on the same data splits, with the same metrics, by the same evaluation script. Zero tolerance for differential advantages.

**Stratification:** comparisons are reported **per input-SNR class** (0, 5, 10, 20 dB), not just globally. Phase 3 showed NeuroFlux has a regime-dependent advantage that a global comparison would hide.

**Dual primary metrics:** `delta_snr` *and* `rpeak_integrity` are both primary. Reporting one without the other obscures the trade-off Phase 3 documented (STDP improves ΔSNR but damages rpeak; passive preserves rpeak perfectly but underperforms on ΔSNR). All baselines must report rpeak — especially the CNN autoencoder, since NeuroFlux's defensible-without-labels claim depends on the gap between CNN-with-labels and trained-NeuroFlux being measurable.

Metrics:
- ΔSNR (primary, stratified by input SNR class)
- R-peak integrity (primary; the clinical metric)
- Mean squared error vs. clean ground truth
- Computational cost (wall time per second of signal, memory footprint)
- Parameter count / training data required (highlight: 0 labels for NeuroFlux/passive vs. labeled set for CNN)

### Deliverables

- `neuroflux-baselines` package with all baseline implementations.
- Unified evaluation script: `evaluate --methods all --dataset mit-bih --stratify-by input_snr`.
- Comparison table (publication-ready) with all methods × all metrics × confidence intervals × input-SNR strata.
- Plots: ROC-style curves for each method, example signals showing before/after, *and* a stratified bar chart (ΔSNR × input-SNR class × method).
- Statistical significance tests (paired Wilcoxon signed-rank across all test subjects, separately per input-SNR class).
- A "passive-vs-trained NeuroFlux" ablation section explicitly quantifying the contribution of STDP.

### Honesty requirements

This phase must be ruthlessly honest. If NeuroFlux underperforms on a metric, that gets reported prominently. The goal is not to "win" against every baseline — the goal is to understand where neuromorphic processing has genuine advantages and where it does not.

**Likely honest findings (revised after Phase 3):**
- At input SNR ≤ 0 dB, trained NeuroFlux likely beats Butterworth/LMS on ΔSNR; at input SNR ≥ 5 dB, classical filters likely win. The defensible NeuroFlux story is "useful in severe-noise regimes where conventional filters cannot adapt."
- **Passive NeuroFlux is expected to beat trained NeuroFlux on rpeak preservation** (Phase 3 §5.6.2 already shows this — rpeak=1.000 untrained vs ≤0.89 trained). This is a direct ablation that quantifies STDP's net effect.
- CNN autoencoder likely beats both NeuroFlux configurations on ΔSNR with labels available; its win shrinks substantially at low input SNR. NeuroFlux's pitch: equivalent clinical-feature preservation without any labeled data.
- Wavelet denoising may excel on transient preservation — NeuroFlux's advantage is online adaptation and zero labeled training data.

Framing these honestly strengthens the proposal rather than weakening it.

### Acceptance criteria

- All seven baselines (including passive NeuroFlux) implemented and tested.
- Unified evaluation produces a single stratified results table with statistical tests per input-SNR class.
- At least one input-SNR regime identified where trained NeuroFlux is superior to classical filters, with a clear mechanistic explanation.
- At least one input-SNR regime identified where classical filters or CNN are superior, with honest analysis.
- The passive-vs-trained NeuroFlux ablation cleanly characterizes what STDP adds.
- The Phase 3 regression bar (`tests/datasets/test_phase3_regression.py`) is not regressed by any Phase 4 work.

---

## Phase 5 — Enhanced interactive dashboard

**Duration:** 2 weeks
**Goal:** Evolve the existing browser demo into a full-featured research and demonstration tool with real data upload, batch experiment management, and publication-quality export.

### Features to add (beyond current demo)

- **File upload:** Drag-and-drop support for EDF, WFDB, CSV formats. Parse and auto-detect signal type.
- **Real dataset demos:** Preloaded MIT-BIH samples accessible via dropdown.
- **Baseline comparison view:** Side-by-side display of NeuroFlux vs. Butterworth vs. LMS on the same input.
- **Export:** Download filtered signals as CSV/EDF, export parameter configurations as YAML.
- **Experiment history:** LocalStorage-based log of past runs with "revisit" capability.
- **Narrative mode:** Guided walkthroughs for first-time users (ECG flicker demo, EEG artifact demo, pattern recognition demo).
- **Mobile responsiveness:** Conference demos often happen on tablets.
- **Accessibility:** Keyboard navigation, screen reader support for all controls.

### Technical upgrades

- **State management:** Introduce Zustand or Jotai for cleaner state logic as complexity grows.
- **WebWorker offloading:** Move simulation to a Web Worker so UI stays responsive even for long runs.
- **WebGPU (stretch):** If scaling to larger networks, evaluate WebGPU for acceleration — saves server costs.
- **API bridge:** Optional backend connection that lets the browser UI drive the Python engine for large experiments (WebSocket-based).

### Deliverables

- Deployed web application (Vercel, Netlify, or GitHub Pages).
- Demo URL distributable to reviewers, conference attendees.
- User guide with annotated screenshots.
- Embedded analytics (privacy-respecting — Plausible or similar) to understand usage patterns.

### Acceptance criteria

- Complete workflow: user uploads an ECG file, runs NeuroFlux, downloads filtered output — entirely in browser, no server.
- Dashboard renders correctly on desktop (Chrome, Firefox, Safari) and tablet.
- Performance: no UI freezing during simulation (worker offload confirmed).

---

## Phase 6 — Hyperparameter optimization

**Duration:** 1 week
**Goal:** Find optimal NeuroFlux configurations per signal type automatically, rather than relying on manual tuning.

### Approach (revised post-Phase-3)

Bayesian optimization over the key hyperparameters using Optuna:

- `network_layers` — **primary lever**. Phase 3 §5.6.3 showed `[32, 32, 32]` outperforms the historical default `[32, 16, 8]` (+1.04 dB ΔSNR at 0 dB / 10 ep). Start the search near uniform-width architectures; avoid input width ≥ 64 without re-tuning the decoder (the `[64, 32, 64]` failure mode collapses rpeak to 0.0-0.4).
- Encoder `max_firing_rate_hz` — controls information capacity at the input stage; not yet swept.
- STDP A+, A−, τ+, τ− — Phase 3 §5.6.2 showed STDP creates a destructive intermediate state at 10 epochs that partially recovers by 50; tighter A−/A+ symmetry might smooth this trajectory.
- Neuron τ_m and threshold offset
- Population decoder smoothing time constant

For each target dataset (ECG, EEG, PPG), run 100 trials. **Joint objective** (revised): maximize the product of `delta_snr` *and* `rpeak_integrity` on the Phase 3 canonical regression config (record 214 / 50 Hz / 0 dB / 10 epochs). The single-metric ΔSNR objective is wrong — Phase 3 showed STDP improves ΔSNR uniformly while damaging rpeak, so optimizing on ΔSNR alone would regress clinical-feature preservation.

**Hard constraint:** the optimized config must not regress the passive (untrained) network's rpeak below 0.85, and must not regress `tests/datasets/test_phase3_regression.py`.

### Deliverables

- Optimized configs for each signal type, checked into the repo.
- Per-signal-type "recipe" documentation: given a new ECG recording, use config X. Given a new EEG, use config Y.
- Meta-analysis: which hyperparameters are stable across signal types vs. signal-specific.
- An explicit comparison of optimized-trained vs. passive (epochs=0) configurations on rpeak, since Phase 3 showed passive achieves rpeak=1.000 — Phase 6 must demonstrate that the joint optimization can recover most of that.

### Acceptance criteria

- Optimized configs outperform hand-tuned defaults by at least 0.5 dB ΔSNR on held-out test data *and* preserve rpeak ≥ 0.85 (i.e., close most of the gap to the untrained 1.000 ceiling).
- Optimization results are reproducible (seed-controlled).

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting to validation set | Use nested cross-validation; report test-set numbers only |
| Optimization budget inadequate | Start with 100 trials per config; increase only if gains are plateauing |

---

## Phase 7 — Documentation, paper draft, and handoff

**Duration:** 2 weeks
**Goal:** Package everything for external consumption — reviewers, collaborators, future contributors.

### Documentation deliverables

- **User guide:** How to install, how to run common experiments, how to interpret results.
- **API reference:** Auto-generated from docstrings (Sphinx autodoc).
- **Architecture documentation:** Deep technical explanation for contributors.
- **Tutorials:** Three end-to-end Jupyter notebooks — ECG denoising, EEG artifact rejection, pattern recognition.
- **FAQ:** Drawn from actual questions asked during Phases 2–5.

### Paper draft

Target venue (revised post-Phase-3):
- **Frontiers in Neuroscience – Neuromorphic Engineering** (open access, mechanism-friendly framing — preferred given the Phase 3 regime-characterization result; see `docs/phase3_findings.md` §6.3)
- IEEE Transactions on Biomedical Engineering (clinical focus — appropriate if Phase 4 establishes a clinical superiority claim in at least one regime; currently lower probability)
- Nature Scientific Reports (broad readership — appropriate if Phase 4 produces a clean cross-method comparison)

**Headline framing (revised post-Phase-3):** not "we built a better filter," but "we characterized a regime in which biologically-inspired unsupervised reconstruction provides amplitude-domain SNR improvement at low input SNR, while passive SNN architecture alone preserves clinical R-peak landmarks with perfect fidelity. The contribution of STDP is bounded and measurable, and the regime where the trained system is preferable to the passive system is precisely characterized."

Paper structure:
1. Abstract
2. Introduction (problem framing, gap, contribution)
3. Related work (existing SNN frameworks, denoising methods)
4. Methods (Phase 1 architecture, Phase 2–3 methodology)
5. Results (Phase 4 comparisons)
6. Discussion (clinical implications, limitations, future work)
7. Reproducibility statement (code, data, configs all public)

### Handoff materials

- **Onboarding guide** for a new developer: get from zero to running an experiment in under 30 minutes.
- **Video walkthrough** (5–10 minutes) of the codebase architecture.
- **Known issues list** with workarounds.
- **Roadmap** for future development (what Phase 8+ could look like).

### Acceptance criteria

- External reviewer (someone not involved in development) can clone the repo, run the quickstart tutorial, and get results in under 15 minutes.
- Paper draft complete and internally reviewed.
- All deliverables public (documentation website deployed, paper preprint on arXiv).

---

## Phase 8 (optional) — Neuromorphic hardware port

**Duration:** 3 weeks
**Goal:** Demonstrate that NeuroFlux, validated in software, can deploy to actual neuromorphic hardware for orders-of-magnitude energy efficiency gains.

**Phase 3 caveat (added post-sweep):** The energy-efficiency claim is only compelling if Phase 6 produces a configuration whose joint (ΔSNR, rpeak) performance is competitive with classical baselines in at least one regime. If Phase 6 cannot do that — i.e., if the trained NeuroFlux cannot recover most of the rpeak ceiling that the passive variant achieves — Phase 8 becomes "low-power deployment of a regime-limited algorithm," which is a weaker contribution than the original Loihi/Xylo pitch. Treat Phase 8 as **strictly contingent on Phase 6 success** and the first phase to descope under time pressure.

### Target platforms (pick one)

| Platform | Pros | Cons |
|---|---|---|
| Intel Loihi 2 | Mature, well-supported, research access via INRC | Requires INRC membership; LAVA framework learning curve |
| SynSense Xylo | Commercial, low-power, on-chip learning | Smaller research community |
| BrainScaleS-2 | European open-access, fast emulation | Remote-only access |
| SpiNNaker2 | Scalable, ARM-based | Still in rollout |

**Recommendation:** Loihi 2 for academic visibility; Xylo for path to commercial embedded deployment.

### Porting challenges

- Weight quantization (typically 8-bit on hardware vs. 32-bit in software).
- Spike timing quantization (hardware operates on fixed timesteps).
- Different STDP implementation on each platform — may need rule translation.
- Limited fan-out / fan-in constraints per neuron core.
- Ensuring software-validated behavior survives quantization.

### Deliverables

- Loihi-deployed NeuroFlux with equivalent performance to software version.
- Power consumption benchmark: microwatts per second of signal processed.
- Comparison paper section: software vs. hardware performance and energy.

### Acceptance criteria

- Hardware deployment achieves within 15% of software SNR performance.
- Measured power consumption at least 100× lower than GPU baseline.

---

## Cross-cutting concerns

These apply across all phases, not to any single one.

### Version control discipline

- Trunk-based development with feature branches merged frequently.
- Every PR requires: passing CI, at least one review, updated tests.
- Semantic versioning for packages (0.1.0 after Phase 1, 1.0.0 after Phase 7).

### Code quality gates

- Type hints required on all public APIs.
- Test coverage minimum: 85% for core package, 70% for others.
- No merge-and-fix-later — broken main branch blocks all work.

### Experiment reproducibility

- Every figure in every report must be traceable to a specific commit + config + seed.
- Experiment tracking (W&B or MLflow) captures full provenance.
- Large datasets versioned with DVC, not Git.

### Security and privacy

- Never commit real patient data, even de-identified, to public repos.
- All biomedical data stored in a separate, access-controlled location.
- IRB / ethics review engaged before any direct human-subject data collection.

### Communication cadence

- Weekly written progress notes (one page, phase/blockers/next steps).
- End-of-phase demo + written report for each phase.
- Stakeholder review at phase transitions to confirm priorities haven't shifted.

---

## Risk register

| Risk | Phase | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| Performance on real data falls short of synthetic | 3 | High | Medium | Early Phase 3 pilot (one dataset, one week) to de-risk before full integration |
| Key contributor leaves mid-project | All | High | Low | Knowledge shared via docs; no single points of failure in core code |
| Dataset licensing blocks deployment path | 3 | Medium | Medium | Start with most permissive datasets; track license at pipeline level |
| Scope creep from interesting research threads | All | Medium | High | Phase-gated discipline; new ideas go to backlog, not current phase |
| Hardware port (Phase 8) fails due to platform limitations | 8 | Low | Medium | Phase 8 is optional; core contribution stands without it |
| Peer review requests experiments not in plan | 7 | Medium | Medium | Leave 20% of Phase 7 capacity unscheduled for review responses |

---

## Success metrics

The project is considered successful if, by end of Phase 7:

1. NeuroFlux is demonstrated on at least one real biomedical dataset with measurable SNR improvement over unfiltered input.
2. Head-to-head comparison with conventional baselines is complete and publicly documented.
3. Research paper is submitted to a peer-reviewed venue.
4. Public GitHub repository has complete documentation, tutorials, and a deployed interactive demo.
5. At least three external researchers have used the framework (measured via GitHub stars, notebook forks, or citations).

Phase 8 completion (hardware deployment) would be a differentiating bonus but is not required for project success.

---

## Execution notes

**Start date assumption:** The plan assumes a dedicated team of 2–3 engineers plus occasional clinician consultation. Scale durations proportionally for smaller teams.

**Parallelization opportunities:**
- Phase 3 (data integration) and Phase 4 (baselines) can run in parallel once Phase 1 is stable.
- Phase 5 (dashboard) can begin in parallel with Phase 3 once Phase 1's API is frozen.
- Phase 7 documentation begins writing in week 1 of Phase 1 — do not defer documentation to the end.

**What to cut if time is short:**
Priority from most to least critical: Phase 0 → 1 → 3 → 4 → 2 → 7 → 5 → 6 → 8. If under severe time pressure, Phases 5 and 6 can be descoped — the browser demo already exists, and manual hyperparameter tuning is adequate for a first paper.

**What not to cut:**
Phases 3, 4, and 7 are the credibility-bearing phases. Real data, honest baselines, and good documentation are what separate a research prototype from a contribution to the field.

---

*End of implementation plan. For architecture details, see the Proposal Document. For code conventions, see CONTRIBUTING.md (produced in Phase 0).*
