# NeuroFlux paper roadmap — the output-ceiling result

**Status:** Active. Supersedes `PLAN.md` Phases 4, 6 and 7 for the purpose of producing a publishable manuscript.
**Framing:** D — *rate-coded SNN reconstruction imposes an input-independent output quality ceiling.*
**Target:** submission-ready manuscript in 8 weeks of steady part-time work.
**Companion documents:** `docs/phase3_findings.md` (what we measured on ECG), `PLAN.md` (original roadmap), `DETAILS.md` (technical spec).

---

## Why this document exists, and why it was rewritten

The first version of this roadmap aimed at an ECG denoising benchmark paper. Direct measurement killed that framing, and the measurement is worth recording because it is the reason we changed course.

**Classical filtering beats NeuroFlux on every noise type tested**, at 0 dB input SNR on synthetic ECG:

| Noise type | Notch + bandpass ΔSNR | NeuroFlux ΔSNR | Gap |
|---|---:|---:|---:|
| Powerline 50 Hz (pure tone) | +18.60 | +1.76 | −16.84 |
| Broadband gaussian | +3.00 | +1.28 | −1.72 |
| Baseline wander 0.3 Hz | +22.79 | +0.29 | −22.50 |
| Muscle artifact 20–100 Hz | +7.25 | +0.82 | −6.43 |
| Electrode motion (burst) | +1.74 | +0.62 | −1.11 |

**And it beats it on adaptation too**, with interference stepping 50 → 60 Hz mid-record:

| Method | Pre-step ΔSNR | Post-step ΔSNR |
|---|---:|---:|
| Fixed notch @50 only (*mistuned* after the step) | +24.39 | **+11.83** |
| Comb notch @50+60 | +24.24 | +18.05 |
| NeuroFlux trained (10 ep) | +1.97 | +2.13 |

Even a mistuned filter wins by 5×. The structural reason: **0.5–40 Hz bandpass is standard ECG preprocessing and excludes 50 Hz, 60 Hz and every harmonic by construction.** Powerline interference sits outside the ECG diagnostic band. The filter that removes it is already in every clinical signal chain for unrelated reasons. There is no problem to solve.

But look again at the NeuroFlux column in the first table: **1.76, 1.28, 0.29, 0.82, 0.62.** Nearly identical across five completely different noise physics. And in the second table, 1.97 before the frequency step and 2.13 after. The output quality is *independent of what is wrong with the input*.

That is not a failed denoiser. That is a **bounded-capacity signal regenerator**, and characterising it precisely is a contribution to neuromorphic architecture rather than to ECG denoising. That is the paper this evidence supports, and it is the paper this roadmap builds.

---

## The result, as currently measured

All figures below were measured in-session on synthetic ECG (5 s, 250 Hz, 50 Hz flicker at 0.5 amplitude), 3 seeds, `[32,16,8]`, 10 STDP epochs unless stated. **"Ceiling" means affine-corrected output SNR** — the reconstruction optimally rescaled by least squares before scoring, which removes the gain/offset confound described in R1.

### Finding 1 — the ceiling is real and flat

| Nominal SNR_in | SNR_out raw (dB) | SNR_out affine (dB) |
|---:|---:|---:|
| −10 | −4.95 ±3.52 | 0.05 ±0.06 |
| −5 | −2.00 ±2.26 | 0.15 ±0.13 |
| 0 | −1.20 ±1.24 | 0.44 ±0.10 |
| 5 | 0.03 ±0.51 | 0.64 ±0.18 |
| 10 | 1.02 ±0.60 | 1.23 ±0.47 |
| 20 | 1.04 ±0.75 | 1.22 ±0.51 |
| 40 | 0.98 ±0.40 | 1.19 ±0.33 |

**Slope d(SNR_out)/d(SNR_in): raw +0.510, affine +0.108.** Output varies 1.18 dB across the entire input range. A slope of 0 is a perfect ceiling; 1 would mean output tracks input. This is the paper's central figure.

It also explains `docs/phase3_findings.md`'s headline: if SNR_out is constant, then ΔSNR = const − SNR_in falls with slope −1 automatically. The "SNN damages clean signals" finding was the ceiling seen from the wrong side.

### Finding 2 — three invariances

| Knob | Range tested | Ceiling change | Capacity theory predicts |
|---|---|---:|---:|
| Output population N | 4 → 64 | **+0.16 dB** | +12 dB |
| Decoder τ | 5 → 160 ms | **−0.23 dB** (non-monotonic, peaks ≈ 20 ms) | +15 dB |
| Learning | 0 / 10 / 50 / 200 epochs | 0.11 → 0.44 dB | — |

Sixteen times the output neurons buys +0.16 dB. τ has a bandwidth/variance optimum rather than monotonic gain. STDP moves it by fractions of a dB. **The ceiling is not set by the decoder, and it is not set by the learning rule.** That is what makes this an architecture claim rather than a NeuroFlux claim.

### Finding 3 — the architecture cannot be scaled out of it

Widening the *input* population fails in both directions:

| N_in | Fixed g_syn = 30 | g_syn normalised 30·32/N_in |
|---:|---|---|
| 8 | 0.14 dB | 0.11 dB |
| 32 | 0.44 dB | 0.44 dB |
| 64 | **0.00 — output constant (saturated)** | **0.00 — output constant (silent)** |
| 256 | **0.00 — output constant** | **0.00 — output constant** |

With fixed gain the downstream layers saturate; with naive 1/N gain they fall below threshold and go silent. Either way the output becomes constant and carries zero information.

This is the mechanism behind the `[64,32,64]` rpeak collapse documented in `docs/phase3_findings.md` §5.6.3, which was observed but never explained. **The architecture has no gain normalisation**, so `g_syn = 30.0` is implicitly calibrated to `N_in = 32` and the network is not scalable without per-configuration retuning.

### Finding 4 — the gate experiment, run and resolved

Finding 3 did not by itself establish that the ceiling is fundamental — both failures were saturation artifacts of an uncalibrated gain. The decisive experiment was to hold **firing-rate statistics** constant while scaling encoder capacity, and re-measure.

**First attempt (global `g_syn`, output layer pinned at 40 Hz) — still confounded.** Output standard deviation collapsed 0.199 → 0.123 → 0.065 → 0.009 → 0 as `N_in` grew. The rate was pinned but the output stopped *modulating*. Cause: `LIFConfig` carries **one global `g_syn` shared by every layer**, while layer 1's fan-in varies with `N_in` and layer 2's is fixed at 16. One gain cannot serve both — layer 1 saturates and the output decouples from the input.

**Second attempt — per-layer gain calibration**, each layer independently binary-searched to hold 40 Hz:

| N_in | g_L1 | g_L2 | rate L1 | rate L2 | ceiling (dB) | out std |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24.90 | 49.04 | 23.2 | 39.9 | 0.66 ±0.19 | 0.2102 |
| 16 | 20.47 | 22.75 | 40.0 | 39.9 | 1.73 ±0.16 | 0.1327 |
| 32 | 10.47 | 31.69 | 39.9 | 44.9 | 1.72 ±0.38 | 0.1064 |
| 64 | 5.28 | 22.75 | 39.9 | 39.6 | 1.72 ±0.26 | 0.1015 |
| 128 | 2.72 | 27.26 | 39.8 | 40.1 | 1.79 ±0.38 | 0.0986 |
| 256 | 1.36 | 22.75 | 40.2 | 39.8 | 1.44 ±0.43 | 0.1008 |

**Two results, and both matter.**

1. **Per-layer calibration fixes the scaling collapse.** Output std holds at ≈ 0.10 across the full range instead of falling to zero; no configuration is degenerate. This is a working fix for the failure mode in Finding 3, and it is a contribution in its own right.
2. **The ceiling survives the fix.** 32× the input population buys **+0.78 dB** (slope +0.117 dB per doubling) against a capacity prediction of +15 dB.

**The ceiling is therefore not a gain-calibration artifact.** It survived the strongest control available: rate-matched, per-layer, non-degenerate, across a 32× capacity range. This closes the question that would have invalidated the framing.

### Finding 5 — the ceiling is not specific to rate coding

| Encoder | Decoder | Ceiling (dB) | Degenerate? |
|---|---|---:|---|
| Poisson | Population | 1.00 ±0.34 | no |
| Poisson | SpikeRate | 0.43 ±0.17 | no |
| Temporal | Population | 0.11 ±0.15 | no |
| Temporal | SpikeRate | 0.14 ±0.20 | no |
| Delta | Population | 0.00 ±0.00 | **YES** |
| Delta | SpikeRate | 0.00 ±0.00 | **YES** |

| Signal | Ceiling (dB) |
|---|---:|
| ECG | 1.00 ±0.34 |
| EEG | 0.08 ±0.06 |
| Sine | 0.08 ±0.07 |

Temporal coding does **not** escape the ceiling — it is *worse*. The claim is therefore broader than "rate coding is the problem" but flatter: the limit holds across both encoders that function, both decoders, and all three signal types. The Delta rows are degenerate at the threshold used and need tuning before they can be reported as evidence either way.

### What remains open

The phenomenon is established. **What binds it is not.** That is now the paper's research content, addressed by M2 (per-stage information localisation) and M3 (analytical bound). The paper stands without them — Findings 1–5 plus the calibration fix is a complete characterisation — but they are what would make it strong rather than merely solid.

---

## Framing statement

> Rate-coded spiking reconstruction pipelines impose an output quality ceiling that is independent of input quality, output population size, decoder integration window, and learning rule. We measure the ceiling, identify the binding constraint, and show that the architecture cannot be scaled past it without explicit gain normalisation — a failure mode that has been observed in the literature as unexplained performance collapse at larger network sizes.

**Audience:** anyone building rate-coded neuromorphic signal-processing pipelines. The contribution is a limitation others will hit, with the mechanism quantified and a diagnostic protocol they can run.

**The ECG work becomes the motivating application, not the claim.** Section IV-A reports honestly that classical filtering wins, and uses that to motivate the question "why is the SNN's output quality so insensitive to its input?"

---

## Honesty requirements

- **Every table reports the affine-corrected figure alongside the raw one**, and states which is primary. A measured +7 to +10 dB of apparent improvement is pure gain/offset artifact (R1); silence on this is not acceptable.
- **Every ceiling measurement reports a degeneracy check.** An affine SNR of exactly 0.00 dB with zero variance means a constant output, not a low score. Report output standard deviation and mean firing rate beside every ceiling number.
- **Negative results appear in the abstract.** The ECG comparison loses. Say so in sentence three, not on page 7.
- **Claims of invariance get confidence intervals**, never "no significant difference" alone.
- **No citation goes in the manuscript that an author has not opened and read.**
- **The `g_syn` confound is disclosed prominently**, including that our first normalisation attempt overcorrected. Reviewers trust papers that show their failed controls.

---

## Stage R0 — Positioning and literature

**Duration:** 3 days
**Goal:** Establish novelty in the *neuromorphic* literature, which is a different search from the one the previous roadmap specified.

### Work

- [x] **Rate-coded SNN information capacity.** Done. No closed-form per-timestep bound found for rate coding, but two adjacent theories apply directly — see the verdict below.
- [x] **SNN scaling and normalisation.** Done, and the answer is yes. Established since 2015.
- [x] **SNN regression / signal reconstruction.** Done. Reconstruction with SNNs exists but is image-dominated and chooses temporal/direct coding almost universally.
- [x] **Unsupervised STDP.** Done. Diehl & Cook 2015 verified as *Front. Comput. Neurosci.* 9:99 (an early search returned a wrong TNNLS reference — do not use it).
- [x] **ECG denoising** baselines sourced.
- [x] Draft the positioning sentence. Drafted in `docs/related_work.md`; it currently promises the M2 result and is not yet earned.

### Outcome — the two verdicts

Full annotated bibliography: **`docs/related_work.md`** (25 entries, 10 read in full).

**Verdict 1 — is the ceiling already characterised? Partially, and it cuts deeper than expected.**
Moreno-Bote et al. (2014, *Nat. Neurosci.* 17:1410) prove Fisher information saturates with
population size whenever differential correlations exist, and state the sufficient condition
directly: *"If the information provided by the outside world about a stimulus is finite, then
information must saturate as we increase the number of neurons coding for it"*, because
*"if neurons share input, the resulting shared variability in their response cannot be averaged out."*

`cli.py:289` tiles one noisy scalar across every input channel
(`np.tile(noisy, (n_input, 1)).T`). That is maximal shared input noise. **Finding 4's
`N_in` result is therefore what the theory predicts, and the roadmap's "+3 dB per doubling"
expectation was wrong** — it assumed independent samples the code never provided. Finding 4 must
not be published in its current framing. See R4 M1a for the experiment that resolves this.

Separately, Castagnetti et al. (2023, SPIDEN) already measured a **~8 dB penalty** for rate coding
in SNN image denoising. The *direction* of our headline claim is published prior art. What is not
published is the systematic invariance characterisation or the mechanism.

**Verdict 2 — is SNN gain normalisation established? Yes, emphatically.**
Diehl et al. (2015, IJCNN) weight normalisation / threshold balancing; Rueckauer et al. (2017)
percentile normalisation; Zheng et al. (2021) tdBN. **`calibrate_gain` is not a contribution.**
Ship it as required infrastructure, cite the prior art, and recover the value by reframing the
`[64,32,64]` collapse as a demonstration of what happens when this known problem is left unsolved
in an unsupervised STDP pipeline, where no reference activation distribution exists to normalise against.

### Deliverables

- `docs/related_work.md` — annotated bibliography, one paragraph per entry, each with the specific number or claim we will cite.

### Acceptance criteria

- ≥ 20 entries, every one opened and read.
- A definitive yes/no on whether the rate-coding capacity ceiling is already characterised in the literature, with citations.
- A definitive yes/no on whether SNN gain normalisation is an established topic.

---

## Stage R1 — Measurement validity

**Duration:** 1 week
**Goal:** Make the ceiling measurable without artifacts. This is now the most important stage in the roadmap, because the entire result is a measurement.

### Work

- [ ] **`metrics/reconstruction.py` — affine-corrected SNR as a first-class metric.**
  - `affine_corrected_snr(clean, recon)` — fit `a·x̂ + b` by least squares, then score. Return the fitted `a` and `b` alongside, because `a ≈ 0` signals a degenerate output.
  - Raw `calculate_snr` stays for continuity. Every table reports both.
  - **Rationale, measured:** an optimal affine rescale of the *unprocessed noisy signal alone* yields +7 to +10 dB apparent improvement. Raw ΔSNR conflates gain mismatch with denoising, in both directions.
- [ ] **`metrics/degeneracy.py` — guard against scoring constant outputs.**
  - `is_degenerate(recon, tol=1e-9)` → true when output std ≈ 0 or fitted gain ≈ 0.
  - Every experiment records `output_std`, `fitted_gain`, `mean_firing_rate_hz` and `silent_neuron_fraction`. The last two already exist in `metrics/spike_stats.py` and are currently unused.
  - **This is not optional.** Findings 3's `0.00 ±0.00` entries would be silently misread as "low ceiling" without it.
- [ ] **`metrics/distortion.py`** — `prd`, `prd1`, `rmse`, `max_cross_correlation`. Needed for the motivating ECG section to speak the field's vocabulary.
- [ ] **`metrics/qrs.py`** — replace the saturated `rpeak_integrity`. It returns 1.000 for the unprocessed noisy signal down to −20 dB SNR with 6× flicker, because `rpeak.py:54-62` sets the prominence floor to `0.5 × std` of each signal *independently*, so it self-normalises. Use `neurokit2.ecg_peaks` scored against MIT-BIH `.atr` annotations at ±150 ms (ANSI/AAMI EC57 convention — verify against the standard text); report Se, PPV, F1 and **RMS timing jitter in ms**.
  - Mark `rpeak_integrity` deprecated in its docstring with a pointer and a one-line reason.
- [ ] Unit tests for every new metric, including a **saturation test**: each metric must produce materially different scores for the identity baseline versus a genuinely processed signal, across the SNR range, separated by more than the seed-to-seed standard deviation.

### Acceptance criteria

- Every new metric demonstrably discriminates, proven by the saturation test.
- No experiment can report a ceiling number without an accompanying degeneracy verdict — enforced in code, not by convention.
- A short section appended to `docs/phase3_findings.md` recording the `rpeak_integrity` saturation finding and its consequence for §5.6.2.
- Four CI gates stay green.

---

## Stage R2 — Statistical validity

**Duration:** 3 days
**Goal:** Make the unit of analysis correct, and make invariance claims defensible.

### The defect

`analysis.py:193` does not select `record_id`, and `_build_phase3_tasks` stamps `signal_type="ecg"` on every record, so all 16 records collapse into 32 groups with `n_seeds = 80`. Between-patient variance is counted as within-config seed noise — pseudo-replication, visible to any reviewer from the reported *n*.

### Work

- [ ] Fix `fetch_raw_runs` to select all 12 columns. One line; also makes `rpeak_integrity` reachable from the analysis layer for the first time.
- [ ] Add `record_id` and `dataset` to the grouping key in `compute_statistical_significance`.
- [ ] `paired_comparison(raw_runs, a, b, stratify_by=...)` implementing the protocol below.
- [ ] **Equivalence testing for the invariance claims.** "The ceiling does not change with N" is a claim of *no effect*, and a non-significant p-value does not establish one. Use **TOST** (two one-sided tests) against a pre-registered equivalence margin — propose ±0.5 dB — or report the bootstrap CI and argue from its width.

### Protocol

```
Unit of analysis   Record (or seed-configuration), not run. Average
                   within unit before testing. State this explicitly.

Invariance claims  TOST equivalence test, margin +/- 0.5 dB, pre-registered.
                   A CI entirely inside the margin supports invariance.
                   p > 0.05 alone does NOT.

Comparisons        Wilcoxon signed-rank, paired, within each stratum.
                   scipy.stats.wilcoxon

Correction         Holm-Bonferroni within each family. State family size.

Effect size        Median paired difference + bootstrap 95% CI
                   (10,000 resamples, BCa). scipy.stats.bootstrap

Reporting          Every table gets an identity/control row.
```

### Acceptance criteria

- `significance_results.csv` shows `n_seeds` equal to configured seeds per record, not seeds × records.
- Group count equals `records × flicker × snr × epochs × tau`.
- A regression test fails if `record_id` is dropped from the grouping key.
- Every invariance claim in the manuscript is backed by an equivalence test, not a null result.

---

## Stage R3 — Ceiling characterisation

**Duration:** 1.5 weeks
**Goal:** Turn the in-session probes into properly seeded, statistically supported results. This is Section IV of the paper.

### C1 — Flatness (the central figure)

Extend Finding 1: input SNR from −20 to +40 dB in 5 dB steps, ≥ 10 seeds, on **synthetic ECG, real MIT-BIH ECG, EEG, and sine**. Report raw and affine, with the fitted slope and its CI.

- [ ] Note the confound found in-session: because `add_gaussian_noise` is applied to the *already-flickered* signal, measured input SNR saturates around −6 dB regardless of the nominal knob. **Sweep flicker amplitude independently, or drop flicker for the capacity experiments**, so the input-SNR axis actually spans the range it claims to.
- [ ] **Produces:** Fig. 1 — SNR_out vs SNR_in, four signal types, slope ≈ 0

### C2 — Invariances

Extend Finding 2 with proper seeds and equivalence tests: output population N (4→64), decoder τ (5→160 ms), training epochs (0→200), network depth (2, 3, 4 layers), STDP τ.

- [ ] **Produces:** Fig. 2 — ceiling vs each knob, with equivalence bands. Table I.

### C3 — Scaling fragility

Extend Finding 3. Map the `(N_in, g_syn)` plane and locate the operating region where the network is neither saturated nor silent. Report mean firing rate and silent-neuron fraction across the plane.

- [ ] **Produces:** Fig. 3 — a 2-D phase diagram of the usable operating region. This is likely the paper's most immediately useful figure for practitioners.

### Acceptance criteria

- Slope of SNR_out vs SNR_in has a CI excluding 1.0 on all four signal types.
- Invariance to N and to learning rule supported by equivalence tests, not null results.
- The operating-region map identifies the saturated and silent boundaries quantitatively.
- Zero ceiling numbers reported without a degeneracy verdict.

---

## Stage R4 — Mechanism

**Duration:** 1.5 weeks
**Goal:** Find what binds the ceiling. M1 is *not* resolved after all — R0 invalidated its framing. M1a below is now the single most important experiment in the project.

### M1a — The shared-input control — **RESOLVED: the ceiling holds**

R0 (Verdict 1) found that Finding 4's `N_in` sweep cannot test what it claimed to test. Every input channel receives the same noisy scalar (`cli.py:289`), so the information about the clean signal is fixed at one sample's worth no matter how large `N_in` grows. Moreno-Bote et al. (2014) prove information *must* saturate under exactly this condition. The measured flatness is the predicted result, not evidence of an architectural ceiling.

This is the experiment that decides what the paper is about. **It has been run.**
Full write-up: **`docs/m1a_findings.md`**. Data: `results/m1a_shared_input/m1a_results.csv`.

**Outcome: the ceiling holds — the pre-registered "more interesting" branch.**
Under `independent` input (N genuinely independent noise realisations, where an ideal
estimator gains +3.01 dB per doubling) the network gained **+0.056 dB per doubling**:
+0.25 dB across a 32× width increase where the ideal gains +15 dB. The shortfall widens
monotonically from 8.7 dB at N=8 to 23.5 dB at N=256. `tiled` and `independent` are
statistically indistinguishable. 0 of 54 runs degenerate.

A second control swept input SNR 0 → 40 dB with no flicker: the ceiling moved ~1 dB
across a 40 dB range. **This also clears Finding 1 of the flicker confound** (action 5
below is discharged for the capacity experiments) and its 40 dB column lands at
1.44–1.47 dB, agreeing with `phase3_findings.md` §5.6.1's independent measurement of
~0.7 dB intrinsic output SNR on a clean signal.

The shared-input / information-limiting-correlations explanation is therefore **falsified**
for this system. The bottleneck is downstream of the input representation. **M2 is now the
centrepiece of the paper**, not a supporting result.

- [x] Replace the tiled input with genuinely independent information. Two variants, both worth running:
  - **(i) Independent noise realisations** — same clean signal, `N_in` independent noise draws. This is the clean information-theoretic control: information about the clean signal now genuinely grows with `N_in`.
  - **(ii) Delay-line embedding** — the `N_in` channels carry successive lags of the single noisy channel. This is the realistic engineering variant and was already planned for R5; promote it here.
- [x] Re-run the `N_in` sweep (8 → 256) under per-layer rate control for both variants. **Done at 3 seeds; ≥ 10 seeds and equivalence tests still owed (R2/R4).**
- [x] **Interpretation was fixed in advance, before running** (pre-registered, per the honesty requirements) — the middle branch is the one that occurred:
  - Ceiling **lifts** toward the ~+3 dB/doubling prediction → information-limiting correlations explain it. The paper becomes a characterisation of where a practical rate-coded pipeline sits against the population-coding bound, plus the design rule that tiling a scalar across an input population buys nothing. Solid and useful.
  - Ceiling **holds** → the shared-input explanation is falsified, something downstream binds, and the paper is substantially more interesting. M2 then becomes the centrepiece.
  - Ceiling lifts **partially** → report the fraction of the bound recovered; the residual gap is the contribution.
- [x] **Produces:** the reframed Fig. 4a, and the decision on the paper's central claim.

### M1b — Rate-controlled capacity scaling — production hardening

The per-layer calibration itself stands (it fixed the degeneracy: `out std` holds ~0.10 where global gain collapsed it to 0). Only its *interpretation* was wrong. Remaining work is to turn the probe into shippable, properly powered code — and note R0 Verdict 2: this is established practice, cite Diehl 2015 / Rueckauer 2017 / Zheng 2021, claim no novelty for it.

- [ ] **Ship `calibrate_gain(network, signal, target_rate_hz, tol)` as library code.** Per-layer geometric bisection on `g_syn`, calibrating layer 0 first then freezing it. The probe overrode `net.neurons[i].config` post-construction; production needs per-layer gain as a first-class concept — extend `NetworkConfig` to accept a list of `LIFConfig`, or add a `gains` parameter to `FeedForwardNetwork`.
- [ ] Re-run the `N_in` sweep (8 → 256) and `max_firing_rate_hz` sweep (25 → 250) at ≥ 10 seeds with equivalence tests.
- [ ] Re-run the `N_out` and τ sweeps **under per-layer rate control** — Findings 2's numbers were taken at fixed global gain and may be partly confounded. This is the one place the existing evidence could still shift.
- [ ] Confirm on **real MIT-BIH ECG**, not only synthetic. All in-session probes used synthetic ECG.

### M2 — Locating the bottleneck

Run only if M1 shows the ceiling flat under rate control. Instrument information content at each stage: input signal → encoded spikes → each hidden layer → output spikes → decoded signal. Estimate mutual information (or an SNR proxy) at every stage and find where it collapses.

- [ ] Per-stage SNR proxy: correlate each layer's population rate against the clean signal.
- [ ] **Produces:** Fig. 4 — information loss per pipeline stage. Names the culprit.

### M3 — Analytical bound

- [ ] Derive the per-timestep capacity of the Poisson encoder from `N_in`, `max_firing_rate_hz` and `dt`. Sanity anchor: at `dt = 4 ms` and 100 Hz, `p_max = 0.4`; at 250 Hz, `p_max = 1.0` and the encoder saturates entirely.
- [ ] Compare the predicted bound against the measured ceiling across the M1 sweep. Agreement validates the mechanism; disagreement is itself a finding.
- [ ] **Produces:** Fig. 5 — predicted vs measured ceiling. Section III theory.

### Acceptance criteria

- A definitive, evidenced answer to whether the ceiling is encoder-bound.
- `calibrate_gain` shipped, tested, and demonstrated to hold firing rate constant across ≥ 8 configurations.
- If DPI holds: predicted and measured ceilings agree within a stated tolerance across the sweep.
- If DPI fails: the binding stage is identified by M2 with per-stage evidence.

---

## Stage R5 — Generality

**Duration:** 1 week
**Goal:** Establish that this is an *architecture class* property, not a NeuroFlux quirk. Without this, it is not an architecture paper.

**The repository is unusually well-placed for this.** Three encoders, two decoders and five metrics are fully implemented, tested, and used by nothing. The audit flagged them as dead code; for this paper they are the generality evidence.

- [ ] **Encoder generality** — repeat C1 with `TemporalEncoder` and `DeltaEncoder`. Rate coding may be the culprit; latency and delta coding are the natural controls. **If the ceiling disappears under temporal coding, that is a headline result**, and it reframes the paper as "rate coding specifically is the problem."
- [ ] **Decoder generality** — repeat with `SpikeRateDecoder`.
- [ ] **Signal generality** — synthetic ECG, real MIT-BIH ECG, EEG, sine. Already supported by the CLI.
- [ ] **Topology generality** — depth 2/3/4, and a non-tiled input where the 32 channels carry genuinely different values (a short delay-line embedding) rather than 32 copies of one scalar. **All 32 input channels currently carry identical values**, so this tests whether the population is doing anything beyond variance reduction.
- [ ] **Produces:** Table II — ceiling across the encoder × decoder × signal × topology grid.

### Acceptance criteria

- The ceiling is demonstrated (or refuted) across ≥ 2 encoders, ≥ 2 decoders, ≥ 3 signal types.
- An explicit statement of scope: which architecture class the claim covers and which it does not.

---

## Stage R6 — Manuscript

**Duration:** 2 weeks

```
I.    Introduction
      Rate-coded SNN pipelines are widely used for signal reconstruction.
      We show they impose an input-independent output quality ceiling.
      Contributions as an explicit bulleted list of 3-4 items.

II.   Related Work                                        -> R0
      A. Rate coding and capacity   B. SNN scaling and normalisation
      C. SNN regression/reconstruction

III.  Methods
      A. Architecture under test (LIF, STDP, Poisson, population decode)
      B. Measurement protocol: affine correction + degeneracy guard  -> R1
      C. Analytical capacity model                                   -> M3
      D. Statistical protocol incl. equivalence testing              -> R2

IV.   Results
      A. Motivating case: ECG denoising, where classical filters win -> the tables above
      B. The ceiling is flat in input quality       -> C1   Fig. 1
      C. Invariance to width, window, learning      -> C2   Fig. 2, Table I
      D. Scaling fragility and the operating region -> C3   Fig. 3
      E. Rate-controlled capacity scaling           -> M1
      F. Where the information is lost              -> M2   Fig. 4
      G. Predicted vs measured bound                -> M3   Fig. 5
      H. Generality across encoders and signals     -> R5   Table II

V.    Discussion
      A. Implications for neuromorphic pipeline design
      B. The case for explicit gain normalisation in SNNs
      C. Limitations and threats to validity

VI.   Conclusion
      Reproducibility statement + repository DOI
```

- [ ] Draft in IEEE `IEEEtran` two-column from the start.
- [ ] All figures vector, colourblind-safe, legible at print size.
- [ ] **Reproducibility subsection.** 97% coverage, seeded determinism, SHA-256 data provenance, snapshot-pinned numerics, public sweep databases. Several IEEE venues award reproducibility badges — claim one. Mint a Zenodo DOI for the exact commit.
- [ ] TechRxiv preprint on submission.
- [ ] Internal review pass against the objection table before submitting.

### Venue

The audience moved from biomedical to neuromorphic. Venue moves with it.

| Venue | Type | Fit | Realism |
|---|---|---|---|
| **IEEE IJCNN** | Conference | SNN architecture studies are core scope | **Primary target** |
| **Neuromorphic Computing & Engineering** (IOP) | Journal, OA | The field's home venue; publishes negative and characterisation results | **Strong** (not IEEE) |
| IEEE BioCAS | Conference | Works if the ECG motivation is kept prominent | Good |
| IEEE TNNLS | Journal | Needs M1/M3 to land cleanly with the analytical bound | Reach — realistic if mechanism confirms |
| Frontiers in Neuroscience (Neuromorphic Eng.) | Journal, OA | Receptive to characterisation studies | Backup |
| IEEE EMBC / JBHI | — | Wrong audience now | Not applicable |

**Path:** IJCNN first. If M1 and M3 produce a clean validated bound, the journal version goes to TNNLS or NCE.

---

## Anticipated reviewer objections

| Objection | Answered in |
|---|---|
| "Your ceiling is just a badly tuned `g_syn`." | §IV-E — rate-controlled scaling, M1. This is *the* objection; M1 exists to answer it. |
| "Affine correction is unusual — are you hiding a scaling failure?" | §III-B — raw and affine both reported, with the +7–10 dB identity artifact shown. |
| "A ceiling of 0.00 dB means your network is broken, not bounded." | §III-B — degeneracy guard reports output std and firing rate for every measurement. |
| "This is one architecture; you cannot claim a class property." | §IV-H — encoder × decoder × signal × topology grid, with explicit scope statement. |
| "Non-significant difference is not invariance." | §III-D — TOST equivalence tests with pre-registered margins. |
| "Why should anyone care about a method that loses to a notch filter?" | §IV-A — that is the motivation, stated plainly, not the claim. |
| "Is this capacity limit already known?" | §II — R0's literature search answers this before we write. |
| "n=80 'seeds' from 16 patients is pseudo-replication." | §III-D — record-level analysis, corrected in R2. |
| "Can I reproduce this?" | §VI — reproducibility statement, Zenodo DOI, public databases. |

---

## Schedule

| Week | Stage | Gate? |
|---|---|---|
| 1 | **R0 literature** · R1 metrics · R2 statistics (parallel — different files) | **Yes — novelty is now the binding risk** |
| 2 | R1 metrics · R3 characterisation begins | — |
| 3 | R3 characterisation (C1, C2, C3) | — |
| 4–5 | R4 mechanism (M1 hardening, M2, M3) | — |
| 6 | R5 generality | — |
| 7–8 | R6 manuscript | — |

R0, R1 and R2 parallelise in week 1 because they touch different files.

**The gate has moved twice.** It was M1 — whether the ceiling survives gain control; that was answered yes. R0 then moved it again: M1's *framing* was invalid, because the tiled input (`cli.py:289`) makes the `N_in` sweep unable to test capacity at all. The gate is now **M1a**, the shared-input control. Novelty is settled — narrower than hoped, still real (see R0 Verdict 1). One experiment now decides the paper's central claim, and it is cheap. If rate-coded reconstruction capacity ceilings are already characterised in the literature, the framing and venue change; if they are not, Findings 1–5 are a paper and the remaining stages make it a good one.

Run R0 first, and do not defer it.

---

## Immediate next actions

**R0 is complete.** `docs/related_work.md` holds the annotated bibliography and both verdicts. It changed the ordering below.

1. ~~Run M1a~~ **Done** — the ceiling holds. See `docs/m1a_findings.md`.
2. ~~Ship `calibrate_gain`~~ **Done** — `network/calibration.py`, per-layer gain is now a first-class concept via `NetworkConfig.layer_gains`.
3. ~~Write `metrics/reconstruction.py` and `metrics/degeneracy.py`~~ **Done**, with the R1 saturation test.
4. **Run M2 — localise the bottleneck.** Now the paper's centrepiece. Per-stage SNR proxy: correlate each stage's representation (encoder output, each layer's population rate, decoder output) against the clean signal, and find where the information dies. The candidates are the Poisson encoder, the LIF layer, and the population decoder.
5. **Fix `fetch_raw_runs` to select all 12 columns.** One line, unblocks R2.
6. **Re-run Findings 2 and 5 under per-layer rate control**, now that calibration exists — the `N_out` and τ numbers were taken at fixed global gain. This is the last place existing evidence could shift.
7. **Raise M1a to ≥ 10 seeds and add equivalence tests**, and confirm on real MIT-BIH.
8. **Finish R0's residual reads** — the seven open items at the end of `related_work.md`, chiefly the npj regression paper (potentially the closest task match) and Kanitscheider 2015.
9. **Tune the Delta encoder threshold** before R5, or report it as out of scope. Its rows are currently degenerate and prove nothing.

---

## What success looks like

At the end of R6 we hold a manuscript whose central claim was tested by the experiment most likely to falsify it, and survived — or did not, and the paper reports what bound the system instead.

We are no longer trying to show that NeuroFlux is a good denoiser. The measurements say it is not, and that question is closed. We are characterising *why* an entire class of rate-coded neuromorphic pipelines cannot exceed a fixed output quality, using a codebase whose determinism, coverage and provenance make the measurement trustworthy.

That is a smaller claim than the original proposal made, and a much more defensible one.
