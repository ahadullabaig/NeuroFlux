# NeuroFlux Hackathon Pivot: From Research Characterization to Adaptive ECG Signal Intelligence

**Context:** Phases 0–3 are complete. Phase 3 produced a regime-characterization result that reframed the project from "we built a better filter" to "we know exactly when unsupervised SNN reconstruction helps vs. hurts." This document lays out how to reroute the project into a hackathon-winning product for the Unisys Innovation Program without discarding the rigorous work already done — instead, using Phase 3's findings as the scientific backbone of a larger, more compelling system.

**Core insight driving the pivot:** Phase 3 proved that no single denoising method works everywhere. The passive SNN preserves R-peaks perfectly but doesn't improve SNR. Trained STDP improves SNR at low input quality but damages R-peaks. Classical filters (Butterworth, LMS) are expected to dominate at moderate-to-high input quality but are rigid and require manual frequency specification. The product is not any one of these filters. The product is a system that knows which one to use, when, and why — and does it without human intervention.

---

## The Pitch (one paragraph — memorize this)

> NeuroFlux is an adaptive ECG signal intelligence system that automatically detects the noise regime of incoming ECG data and routes it through the optimal denoising pipeline. Unlike fixed filters that work well in one regime and fail silently in others, NeuroFlux uses a learned noise-regime classifier backed by empirical evidence from 2,560+ validated runs on real patient data to select the right processing strategy per signal segment — from passive SNN architecture for R-peak-critical clinical analysis, to STDP-trained reconstruction for severely degraded signals, to classical notch filtering for routine recordings. No manual tuning, no labeled data, no per-patient configuration. The system doesn't just denoise — it tells you *how confident it is* in the output and *why it chose that strategy*, giving clinicians transparent, trustworthy signal processing for the first time.

---

## Why This Pivot Works for the Hackathon Theme

The theme emphasizes five things repeatedly: systemic thinking, adaptive technologies, real-world impact, scalability, and AI/ML innovation. Map each to what the pivoted project delivers:

| Theme keyword | What you deliver |
|---|---|
| **Systemic thinking** | The entire project is built on the insight that denoising is not one problem — it's a regime-dependent problem that requires understanding the interconnections between noise characteristics, signal morphology, and processing strategy. |
| **Adaptive technologies** | The routing layer adapts per-segment, per-patient, per-noise-regime. No static configuration. |
| **Real-world impact** | ECG misdiagnosis from powerline interference is a documented clinical problem. You tested on real MIT-BIH patient data, not toy examples. |
| **Scalability** | Classical filters are microseconds. The SNN runs in under 2 seconds for 10,000 timesteps. The noise classifier is a lightweight feature extractor. This runs on edge hardware. |
| **AI/ML innovation** | You're combining unsupervised SNN learning (STDP), classical signal processing, and a noise-regime classifier into a single adaptive system. That's exactly the "synergy between Generative AI and Classical AI techniques" the theme calls for (substitute "neuromorphic AI" for "generative AI" — the principle is the same). |

Additionally, the theme explicitly lists "Health Ecosystems" and "AI-powered diagnostics" as example innovation areas. ECG signal quality is upstream of every cardiac AI diagnostic — bad signal in, bad diagnosis out. Your system sits at that critical junction.

---

## Development Plan: Phases 3A through 3D, then Phase 5

**Do not proceed to the original Phase 4 as written in PLAN.md.** Phase 4 was designed as a rigorous academic baseline comparison for a research paper. You need a subset of Phase 4's work (implementing 2-3 baselines) but wrapped in a product framing, not a paper framing. The following sub-phases replace Phase 4 for hackathon purposes. Think of them as Phase 3A–3D: everything builds on the completed Phase 3 foundation. After 3C delivers the adaptive routing engine, Phase 5 (Enhanced Interactive Dashboard) from the original PLAN.md follows as the frontend — built in full scope, not scoped down.

**Estimated total effort:** 3.5–5 days for Phases 3A–3D (backend + routing + pitch prep), then 2 weeks for Phase 5 (full dashboard as specified in PLAN.md). If the hackathon deadline is tight, 3A–3D gives you a working backend and pitch materials; Phase 5 gives you the visual product.

---

### Phase 3A — Implement Minimal Baseline Filters (1–1.5 days)

**Goal:** Get two classical baselines running on the same MIT-BIH evaluation pipeline so you have a real, empirical comparison — not hypothetical claims.

**Why only two baselines:** The hackathon doesn't need seven baselines with Wilcoxon signed-rank tests. It needs a Butterworth notch (what hospitals actually use) and one adaptive filter (LMS) to show the regime-dependent story visually. Two baselines plus passive NeuroFlux plus trained NeuroFlux gives you four methods on one chart — enough to tell the story without drowning in data.

#### Baseline 1: Butterworth Notch Filter

This is the clinical default. Every bedside monitor uses some variant of this. Implementation is trivial:

```python
from scipy.signal import iirnotch, filtfilt

def butterworth_notch(signal, freq_hz, fs, quality_factor=30.0):
    b, a = iirnotch(freq_hz, quality_factor, fs)
    return filtfilt(b, a, signal)
```

That's it. Run it on all 16 records × 4 input SNRs × 2 flicker frequencies × 5 seeds using the same `make_pair` function Phase 3 uses for ground truth generation. Compute `delta_snr` and `rpeak_integrity` with the same metrics. Store results in a SQLite DB with the same schema as `phase3_full_sweep.db` but with `method="butterworth_notch"` as an additional column.

The critical detail: the notch filter needs to know the interference frequency (50 or 60 Hz). In the sweep, you know it because you injected it. In the product framing, the noise-regime classifier will estimate it. For now, pass the known frequency.

#### Baseline 2: LMS Adaptive Filter

LMS is the "current best practice" for adaptive noise removal when you don't have a reference channel. The implementation without a reference channel uses a delayed version of the noisy signal as the reference (self-referencing LMS):

```python
import numpy as np

def lms_filter(signal, mu=0.01, filter_order=32):
    n = len(signal)
    w = np.zeros(filter_order)
    output = np.zeros(n)
    for i in range(filter_order, n):
        x = signal[i - filter_order:i][::-1]
        y = np.dot(w, x)
        e = signal[i] - y
        w += 2 * mu * e * x
        output[i] = e
    return output
```

This is slower than the notch but still fast. Run it on the same evaluation matrix. LMS without a reference channel is a direct apples-to-apples comparison with NeuroFlux (both are unsupervised, single-channel).

#### Evaluation Runs

Reuse the existing Phase 3 infrastructure. The simplest approach: write a `run_baseline_experiment` function in a new module (`packages/baselines/src/neuroflux_baselines/filters.py`) that takes the same `(record_id, flicker_freq, input_snr, seed)` parameter tuple as `run_single_experiment`, calls `make_pair` to get the same `(clean, noisy)` pair, applies the baseline filter instead of the SNN, and computes the same three metrics. Run it with `ProcessPoolExecutor` the same way the Phase 3 sweep did.

You need runs for: 16 records × 2 flicker frequencies × 4 input SNRs × 5 seeds × 2 baselines = 1,280 runs. At milliseconds per run (these filters are fast), this completes in under a minute.

#### Deliverable

A combined results database or CSV with columns: `method` (butterworth_notch, lms, passive_neuroflux, trained_neuroflux_10ep, trained_neuroflux_50ep), `record_id`, `flicker_freq`, `input_snr`, `seed`, `delta_snr`, `rpeak_integrity`. This is the empirical foundation for the entire hackathon demo.

#### What you expect to find (based on Phase 3 data)

Be prepared for these likely outcomes:

| Input SNR | Butterworth notch | LMS | Passive NeuroFlux | Trained NeuroFlux (10 ep) |
|---|---|---|---|---|
| 0 dB | Moderate ΔSNR, high rpeak | Moderate ΔSNR, moderate rpeak | Very negative ΔSNR, **perfect rpeak (1.000)** | Small positive ΔSNR (+0.7), moderate rpeak |
| 5 dB | Good ΔSNR, high rpeak | Good ΔSNR, moderate rpeak | Very negative ΔSNR, **perfect rpeak** | Negative ΔSNR, moderate rpeak |
| 10 dB | Very good ΔSNR, high rpeak | Good ΔSNR, high rpeak | Very negative ΔSNR, **perfect rpeak** | Negative ΔSNR, lower rpeak |
| 20 dB | Excellent ΔSNR, high rpeak | Very good ΔSNR, high rpeak | Very negative ΔSNR, **perfect rpeak** | Very negative ΔSNR, lower rpeak |

The story this tells: Butterworth wins on ΔSNR almost everywhere. But passive NeuroFlux wins on rpeak everywhere. And at 0 dB, trained NeuroFlux is the only method delivering positive ΔSNR improvement in a fully unsupervised, no-frequency-knowledge setting. Each method has a regime. That's the product thesis.

If the actual numbers don't match these predictions, adjust the narrative accordingly. Do not fabricate or cherry-pick. The honest version is always more defensible under questioning.

---

### Phase 3B — Build the Noise-Regime Classifier (1–1.5 days)

**Goal:** Build a lightweight classifier that takes a raw ECG segment and outputs a noise regime label (clean, mild, moderate, severe) plus an estimated interference frequency, without any reference signal.

This is the intelligence layer that turns "four separate filters" into "one adaptive system." It's also the novel engineering contribution the hackathon judges will care about — the filters are commodity; the routing is the innovation.

#### Feature Extraction

The classifier doesn't need deep learning. Four simple signal-domain features are sufficient to estimate noise regime:

1. **Spectral peak ratio at 50/60 Hz.** Compute the power spectral density (Welch's method), measure the peak power in narrow bands around 50 Hz and 60 Hz relative to the broadband power. High ratio → strong powerline interference. The dominant peak also tells you whether it's 50 Hz or 60 Hz interference — this replaces the manual frequency input the Butterworth notch needs.

2. **Broadband noise floor.** Median power in the 80–120 Hz band (above typical ECG content, below the Nyquist for 250 Hz sampling). High floor → high Gaussian noise → low input SNR.

3. **QRS SNR proxy.** Ratio of peak amplitude in the 5–15 Hz band (where QRS energy concentrates) to the 45–55 Hz band. This is a fast proxy for input SNR without needing a clean reference.

4. **Baseline wander magnitude.** Power below 1 Hz relative to total power. Not directly relevant to flicker, but important for routing — heavy baseline wander may need a high-pass pre-filter before any denoising.

These four features can be computed in under 10 ms for a 10-second segment using `scipy.signal.welch`. No training data needed for the classifier itself — you can set thresholds empirically from the Phase 3 sweep data.

#### Regime Classification Logic

Define four regimes with routing decisions:

```
SEVERE  (estimated SNR ≤ 0 dB):
  → Primary: Trained NeuroFlux (10 epochs, τ=10ms)
  → Rationale: only method with positive ΔSNR in this regime
  → Confidence flag: LOW (STDP damages rpeak; warn clinician)

MODERATE (estimated SNR 1–10 dB):
  → Primary: Butterworth notch at detected frequency
  → Fallback: LMS if frequency detection is ambiguous
  → Rationale: classical filters dominate here

MILD (estimated SNR 10–20 dB):
  → Primary: Butterworth notch (light touch)
  → Rationale: signal is already good; minimal intervention needed

CLEAN (estimated SNR > 20 dB):
  → Primary: Passthrough (no filtering) OR Passive NeuroFlux
  → Rationale: filtering can only hurt; passive NeuroFlux preserves landmarks
  → Confidence flag: HIGH
```

Additionally, every regime should have the option to run Passive NeuroFlux in parallel as a "clinical landmark preservation" channel. The output of the system is then two things: the denoised signal (from the routed method) and a separate R-peak detection channel (from passive NeuroFlux, which has rpeak=1.000 regardless of regime).

#### Implementation

Create a `NoiseRegimeClassifier` class in `packages/baselines/src/neuroflux_baselines/classifier.py`:

```python
class NoiseRegimeClassifier:
    def __init__(self, fs: float = 250.0):
        self.fs = fs

    def classify(self, signal: np.ndarray) -> NoiseRegimeResult:
        """Returns regime label, estimated SNR, detected frequency, confidence."""
        features = self._extract_features(signal)
        regime = self._classify_regime(features)
        return NoiseRegimeResult(
            regime=regime,  # SEVERE / MODERATE / MILD / CLEAN
            estimated_snr_db=features.snr_proxy,
            detected_flicker_hz=features.dominant_flicker_freq,
            confidence=self._compute_confidence(features),
            recommended_method=self._route(regime),
        )
```

The confidence score is important for the demo and the pitch. It communicates "we don't just process — we tell you how much to trust the output." That's a differentiator over blind filtering.

#### Validation

Run the classifier on every `(record_id, flicker_freq, input_snr, seed)` combination from Phase 3A's evaluation matrix. For each, check: does the classifier's recommended method match the method that actually produced the best ΔSNR and rpeak in the Phase 3A empirical results? Report routing accuracy as a percentage. Even 70–80% routing accuracy is a strong result for a threshold-based classifier with no training data.

---

### Phase 3C — Build the Adaptive Routing Engine (1 day)

**Goal:** Wire the classifier from 3B to the filters from 3A and the SNN from Phases 1–3 into a single `AdaptiveDenoiser` pipeline that takes a raw ECG signal and returns a denoised signal, the processing method used, a confidence score, and an R-peak preservation channel.

#### Architecture

```
Raw ECG segment (e.g., 10 seconds at 250 Hz)
         │
         ▼
┌─────────────────────┐
│  NoiseRegimeClassifier  │──→ regime label + confidence + detected freq
└─────────────────────┘
         │
         ├── SEVERE ──→ Trained NeuroFlux (10 ep, τ=10ms) ──┐
         ├── MODERATE ─→ Butterworth notch @ detected freq ──┤
         ├── MILD ─────→ Butterworth notch (light Q) ────────┤
         └── CLEAN ────→ Passthrough ────────────────────────┤
                                                              │
                                                              ▼
                                                     Denoised signal
                                                              │
         ┌────────────────────────────────────────────────────┤
         │                                                    │
         ▼                                                    ▼
  Passive NeuroFlux                                   Final Output:
  (always runs in parallel)                           - denoised signal
         │                                            - method used
         ▼                                            - confidence score
  R-peak timing channel                               - R-peak locations
  (rpeak = 1.000 guaranteed)                          - noise regime label
```

The key design decision: passive NeuroFlux runs in parallel on every segment regardless of routing, because its R-peak preservation is perfect and costs very little compute (no training loop, just one forward pass). The system always outputs two channels: the amplitude-optimized denoised signal (from whatever the router chose) and the timing-optimized R-peak channel (from passive NeuroFlux). A downstream diagnostic system can use the R-peak channel for rhythm analysis and the denoised channel for morphology analysis.

#### Implementation

```python
class AdaptiveDenoiser:
    def __init__(self, snn_config, fs=250.0):
        self.classifier = NoiseRegimeClassifier(fs=fs)
        self.snn_trained = ...      # Pre-trained NeuroFlux (or train on-the-fly)
        self.snn_passive = ...      # Untrained NeuroFlux (epochs=0)
        self.fs = fs

    def process(self, raw_signal: np.ndarray) -> DenoiserResult:
        regime = self.classifier.classify(raw_signal)

        # Route to appropriate method
        if regime.regime == "SEVERE":
            denoised = self._run_trained_neuroflux(raw_signal)
        elif regime.regime in ("MODERATE", "MILD"):
            denoised = butterworth_notch(raw_signal, regime.detected_flicker_hz, self.fs)
        else:
            denoised = raw_signal  # passthrough

        # Always run passive NeuroFlux for R-peak channel
        rpeak_channel = self._run_passive_neuroflux(raw_signal)

        return DenoiserResult(
            denoised=denoised,
            rpeak_channel=rpeak_channel,
            method_used=regime.recommended_method,
            regime=regime.regime,
            confidence=regime.confidence,
            estimated_input_snr=regime.estimated_snr_db,
            detected_flicker_hz=regime.detected_flicker_hz,
        )
```

#### End-to-End Validation

Run `AdaptiveDenoiser.process()` on the full Phase 3 evaluation matrix (16 records × 4 SNRs × 2 flicker freqs × 5 seeds = 640 unique signal pairs). For each, compare the routed output's ΔSNR and rpeak against every individual method's output. Compute how often the router picks the best or near-best method. Report the improvement of adaptive routing over any single fixed method.

The key number to have ready for judges: "Across N runs on real patient data, adaptive routing matched or exceeded the best fixed method X% of the time, while no single fixed method exceeded Y% across all regimes."

---

### Phase 3D — Pitch Materials and Stress-Testing (0.5–1 day)

**Goal:** Prepare supporting materials and anticipate judge questions.

#### Slide Deck (5–7 slides maximum)

1. **Problem** — ECG noise → misdiagnosis. One stat, one image of a noisy vs. clean ECG.
2. **Why current solutions fail** — Butterworth notch is fixed; fails silently at low SNR. One chart showing Butterworth's ΔSNR collapsing at 0 dB.
3. **Our approach** — Adaptive routing architecture diagram. The four-method roster.
4. **Key finding** — The Phase 3 regime characterization table. "2,560 runs on 16 real patients proved no single method works everywhere."
5. **Demo** — Live walkthrough of the Phase 5 dashboard. This is the centerpiece.
6. **Results** — Adaptive routing accuracy. "Matches or exceeds best fixed method X% of the time."
7. **Roadmap** — Multi-lead input, PTB-XL validation, edge deployment, EEG/PPG extension.

#### Anticipated Judge Questions and Prepared Answers

**"How is this different from just using a notch filter?"**
"A notch filter works great when the signal is moderately noisy and you know the interference frequency. It fails at severe noise levels (we measured this — here's the data) and it can't detect the frequency automatically. Our system includes the notch filter as one tool in a larger adaptive pipeline, and it adds capabilities the notch doesn't have: automatic noise regime detection, unsupervised deep-noise handling via SNN, and guaranteed R-peak preservation via the passive neural channel."

**"What's your SNR improvement?"**
"It depends on the regime — and that's the whole point. At 0 dB input SNR, our SNN achieves +0.7 dB averaged across 16 patients, which is the only positive result among unsupervised methods. At 20 dB, the Butterworth notch achieves the best result and our system routes to it automatically. The adaptive system's value isn't in any single number — it's in always picking the right tool."

**"Isn't +0.7 dB a tiny improvement?"**
"In absolute terms, yes. In context: at 0 dB input SNR, the signal is nearly unreadable. Any positive improvement without labeled data or manual tuning is clinically meaningful because the alternative is discarding the recording entirely. And the R-peak preservation (0.77 integrity) means rhythm analysis is still possible even on severely degraded input. The combination of modest SNR improvement + landmark preservation + zero labels is the contribution."

**"Can this run in real time?"**
"The noise classifier runs in under 10 ms. The Butterworth notch runs in microseconds. The passive NeuroFlux forward pass takes about 2 seconds for a 10-second segment at 250 Hz. The trained NeuroFlux path (used only in the severe regime, which is rare in practice) takes 10–20 seconds with training. For monitoring applications, the system can process in sliding windows with the fast paths (classifier + notch + passive) running in real time and the SNN path running asynchronously when severe noise is detected."

**"Why not just use a deep learning model?"**
"A CNN autoencoder would likely beat our SNR numbers with enough labeled training data. The key difference: our system requires zero labeled data. You don't need a cardiologist to annotate thousands of clean/noisy pairs. You plug it into a new hospital's ECG system and it works on day one. That's the scalability argument — labeled biomedical data is expensive and scarce, especially in underserved clinical settings where this kind of tool matters most."

**"Have you validated on data other than MIT-BIH?"**
Be honest. "Not yet. MIT-BIH is the standard benchmark for ECG denoising research and was our Phase 3 target. Our roadmap includes PTB-XL (21,000+ records) for large-scale validation. The noise injection methodology (synthetic flicker on real morphology) is a known limitation — real-world powerline interference has harmonics and amplitude modulation that pure sinusoidal injection doesn't capture. We're transparent about this boundary."

---

### Phase 5 — Enhanced Interactive Dashboard (from PLAN.md, full scope)

**Duration:** 2 weeks (as originally specified in PLAN.md)
**Goal:** Build the full-featured web dashboard specified in PLAN.md Phase 5, now augmented with the adaptive routing system from 3C as its centerpiece.

Phase 5 was always planned as the frontend layer. The difference now is that it has a much stronger backend to showcase. Instead of just visualizing the SNN in isolation, the dashboard demonstrates the complete adaptive denoising system — regime classification, method routing, dual-channel output, and baseline comparisons — all backed by real patient data.

#### What changes from the original PLAN.md Phase 5 spec

The original Phase 5 spec lists these features: file upload (EDF, WFDB, CSV), real dataset demos (preloaded MIT-BIH), baseline comparison view, export (CSV/EDF/YAML), experiment history, narrative mode, mobile responsiveness, accessibility, WebWorker offloading, optional WebGPU, and an API bridge to the Python engine.

All of that still applies. What gets added on top:

**Adaptive routing as the primary interaction model.** The dashboard's main view is not "run NeuroFlux on this signal." It's "feed this signal to the adaptive system and watch it classify the noise regime, pick a method, and produce dual-channel output." The user sees the regime classification happening, the routing decision being made, and the comparison of what each method would have produced. This is the demo flow that wins the hackathon — it tells the story visually.

**Four-panel layout for the adaptive system.** These four panels should be the default landing view:

- **Panel 1 — Signal Input.** Show a real MIT-BIH ECG segment. Let the user pick a record and an input SNR level (slider from 0 to 20 dB). When they adjust, inject the corresponding flicker and noise. Show the clean reference (faded) and the noisy signal (bold).

- **Panel 2 — Regime Classification.** As soon as the noisy signal is generated, display: detected regime (SEVERE / MODERATE / MILD / CLEAN), estimated input SNR, detected flicker frequency, confidence score. Visual color coding: red for SEVERE, yellow for MODERATE, green for MILD/CLEAN.

- **Panel 3 — Method Comparison.** Show four stacked signal traces: Butterworth notch output, LMS output, Trained NeuroFlux output, Passive NeuroFlux output. Each labeled with its ΔSNR and rpeak score. Highlight the one the router selected. Overlay detected R-peaks on each trace so the audience can visually see which methods preserve peak timing.

- **Panel 4 — Adaptive Output.** Show the final adaptive output: the routed denoised signal plus R-peak markers from the passive channel. Display the method used, the confidence, and a sentence explaining why that method was chosen (e.g., "Input SNR estimated at -2 dB. Severe noise detected. Routing to trained NeuroFlux for optimal amplitude recovery. R-peak timing sourced from passive SNN channel for clinical accuracy.").

**Key visual moments to design for:**

- Switching from 20 dB (Butterworth selected, green confidence) to 0 dB (NeuroFlux selected, yellow confidence) — the system adapts visibly.
- Showing that R-peak markers stay stable across all noise levels because the passive channel always delivers them.
- Showing a side-by-side where the Butterworth output at 0 dB is clearly worse than NeuroFlux, and the NeuroFlux output at 20 dB is clearly worse than Butterworth. Neither method alone is sufficient. The adaptive system picks the winner each time.

**Pre-computation strategy.** Pre-compute all results from Phases 3A/3B/3C and load them from the database at demo time. Do not run the SNN live during the demo — it takes 10+ seconds per segment and dead air kills presentations. Pre-compute every combination and look up results instantly. The classifier and the Butterworth filter are fast enough to run live if you want the "watch it analyze in real time" effect.

#### Technical upgrades (unchanged from PLAN.md)

- **State management:** Zustand or Jotai for cleaner state logic as complexity grows.
- **WebWorker offloading:** Move simulation to a Web Worker so UI stays responsive.
- **WebGPU (stretch):** Evaluate for larger networks if needed.
- **API bridge:** WebSocket connection to the Python backend for driving the `AdaptiveDenoiser` from the browser. This is the critical integration point — the React frontend sends a signal segment to the Python backend via WebSocket, the backend runs `AdaptiveDenoiser.process()`, and the frontend renders the result. Pre-computed results can be cached client-side for instant playback; live processing is available for uploaded files.

#### Acceptance criteria (from PLAN.md, plus adaptive system)

- Complete workflow: user uploads an ECG file, the adaptive system classifies the regime, routes to the best method, and displays dual-channel output — entirely functional.
- Dashboard renders correctly on desktop (Chrome, Firefox, Safari) and tablet.
- Performance: no UI freezing during simulation (worker offload confirmed).
- The four-panel adaptive view is the default landing experience.
- Switching between noise levels visibly demonstrates the regime-dependent routing.

---

## Timeline Summary

| Sub-phase | Duration | Deliverable | Depends on |
|---|---|---|---|
| **3A** — Baseline filters | 1–1.5 days | Butterworth + LMS running on MIT-BIH evaluation matrix; combined results DB | Phase 3 complete |
| **3B** — Noise regime classifier | 1–1.5 days | `NoiseRegimeClassifier` with spectral feature extraction; routing accuracy validated | Phase 3A (needs baselines to validate routing) |
| **3C** — Adaptive routing engine | 1 day | `AdaptiveDenoiser` end-to-end pipeline; improvement stats over fixed methods | Phase 3A + 3B |
| **3D** — Pitch prep | 0.5–1 day | Slide deck, Q&A preparation, rehearsal | Phase 3C |
| **Phase 5** — Interactive dashboard | 2 weeks | Full-featured web UI with adaptive routing as centerpiece (per PLAN.md spec) | Phase 3C |
| **Total (3A–3D)** | **3.5–5 days** | Backend + routing + pitch materials | |
| **Total (3A–3D + Phase 5)** | **3.5–5 days + 2 weeks** | Complete product with frontend | |

Phases 3A and 3B can partially overlap if two team members work in parallel (one on filters, one on the classifier). Phase 5 frontend scaffolding can begin while 3C is being finalized — the four-panel layout and component structure don't depend on the backend being complete. Phase 3D (pitch prep) can happen in parallel with early Phase 5 work.

Note that 3A–3D give you a fully functional backend and a rehearsed pitch. If the hackathon deadline falls before Phase 5 is complete, you can demo the adaptive system via pre-computed results, terminal output, or a minimal notebook — the backend works regardless of the frontend. Phase 5 turns it into a polished, interactive product.

---

## Success Criteria

The project is hackathon-ready when all of the following are true:

1. You have empirical ΔSNR and rpeak numbers for four methods (Butterworth, LMS, Passive NeuroFlux, Trained NeuroFlux) across all four input SNR regimes on real MIT-BIH data.
2. The noise-regime classifier correctly routes to the best or near-best method at least 70% of the time on the evaluation matrix.
3. The `AdaptiveDenoiser` pipeline runs end-to-end and produces dual-channel output (denoised signal + R-peak channel) for any input segment.
4. Every team member can answer the five anticipated judge questions above without hesitation.
5. No claim in the pitch deck is contradicted by the actual data.
6. Phase 5 dashboard is functional with the four-panel adaptive view as the default landing experience, backed by pre-computed results for instant playback.

Items 1–5 are achievable with Phases 3A–3D alone (3.5–5 days). Item 6 requires Phase 5 (additional 2 weeks). If the hackathon deadline only allows 3A–3D, you can demo the adaptive system via the Python CLI, a notebook, or a minimal Streamlit wrapper as a stopgap while Phase 5 proceeds.

If the routing accuracy is below 70%, simplify the classifier thresholds or reduce to three regimes instead of four.

---

## After the Hackathon

Regardless of the outcome, the work done in 3A–3D + Phase 5 feeds directly back into the research trajectory:

- Phase 3A's baselines become Phase 4's starting point (add wavelet + CNN to complete the full comparison framework specified in PLAN.md).
- Phase 3B's classifier is a genuinely novel component not in the original PLAN.md — it could become its own section in the paper.
- Phase 3C's adaptive routing concept is a contribution beyond what most ECG denoising papers offer (they compare methods; you route between them).
- Phase 5's dashboard is already the original PLAN.md Phase 5, so it feeds directly into Phase 6 (hyperparameter optimization) and Phase 7 (documentation and paper) without rework.

Nothing is wasted. The hackathon pivot accelerates the research roadmap rather than diverging from it. The sequence after the hackathon resumes at Phase 4 (completing the remaining baselines — wavelet, CNN autoencoder — and running the full stratified statistical comparison), then continues through Phase 6–8 as specified in PLAN.md.
