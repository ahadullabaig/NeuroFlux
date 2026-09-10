# Related work — R0 deliverable

**Status:** first pass complete. **31 entries** — 10 read in full, 6 verified at abstract level,
15 still metadata-only. See *Verification status* below for the per-entry marks.

This document exists to answer two questions the roadmap made blocking, and to supply the
citations the manuscript will hang on. It is not a survey. Every entry ends with the specific
number or claim we intend to cite it for.

---

## Verification status

Entries are marked with how far they were actually opened. The roadmap's acceptance criterion
says "every one opened and read"; that is **not yet true**, and the marks say where the gaps are.

| Mark | Meaning |
|---|---|
| **[read]** | Full text obtained and read. Claims below are quoted or paraphrased from the source. |
| **[abs]** | Publisher page / abstract read directly. Bibliographic data verified. Body not read. |
| **[meta]** | Known only from search results. **Must be opened before citing.** |

**10 read, 6 abs, 15 meta.** The roadmap's "≥ 20 entries, every one opened and read" criterion is
met on count (31 ≥ 20) but **not** on depth: 15 entries are still search-level only. None of those
15 is load-bearing for the two verdicts below — every claim in Verdict 1 and Verdict 2 rests on a
**[read]** or **[abs]** source. The residual reads are listed at the end of this document.

---

## Verdict 1 — Is the rate-coding capacity ceiling already characterised?

**Partially, and this is the single most consequential result of R0.**

Split the question in three:

**(a) Does a general theory predict information saturation with population size? — YES, decisively.**
Moreno-Bote et al. (2014) prove that Fisher information saturates as the number of neurons
*N* → ∞ whenever *differential correlations* are present, and that these are the *only*
correlations that can do so. Their asymptotic result: information follows `I_N = I₀ / (1 + ε·I₀)`,
which "saturates at 1/ε as N goes to infinity". Critically for us, they state the sufficient
condition in plain terms:

> "If the information provided by the outside world about a stimulus, s, is finite, then
> information must saturate as we increase the number of neurons coding for s."

and name the mechanism:

> "If neurons share input, the resulting shared variability in their response cannot be averaged out."

Kanitscheider et al. (2015) then show this arises specifically from **noise injected at the
sensory periphery and propagated forward** — which is exactly NeuroFlux's setup.

**This directly implicates our experimental design.** `cli.py:289` reads
`time_varying_noisy = np.tile(noisy, (n_input, 1)).T` — every one of the `n_input` channels
carries the *identical* noisy scalar at each timestep. That is maximal shared input noise: the
information about the clean signal in the tiled input is the information in a single noisy
sample, and is independent of `n_input`. Under Moreno-Bote's theorem, information *must*
saturate. The measured +0.117 dB per doubling of `N_in` is not an anomaly — **it is what the
theory predicts.** The roadmap's "+3.0 dB per doubling" DPI expectation assumed independent
samples the code never provided.

**(b) Has rate coding specifically been shown to cost reconstruction quality? — YES, but narrowly.**
Castagnetti et al. (2023, SPIDEN) ablated encoding schemes for image denoising and found their
rate-coded variant performed "almost 8 dB lower compared to the other SNNs", attributing it to
"the stochastic nature of the generation process". That is one ablation row inside a supervised
deep-SNN paper — not a characterisation — but it is direct, quantitative, published prior evidence
for the *direction* of our headline claim.

**(c) Has the ceiling been systematically characterised — its invariance to width, τ, learning,
encoder, decoder, and signal type, with a mechanism? — NO.**
Nothing found does this. This is the gap the paper can occupy.

---

## Verdict 2 — Is SNN gain normalisation an established topic?

**YES, emphatically. This closes off `calibrate_gain` as a novelty claim.**

There is a mature line of work on exactly the problem our per-layer calibration solves:

- **Diehl et al. (2015, IJCNN)** introduced weight normalisation and threshold balancing to
  regulate firing rates in converted SNNs — the two are shown to be equivalent (scale weights and
  set threshold to 1, or leave weights and set threshold to the normalisation factor).
- **Rueckauer et al. (2017)** made it robust with percentile-based normalisation (p ∈ [99.0, 99.999]).
- **Zheng et al. (2021)** brought it into directly-trained SNNs as threshold-dependent
  BatchNorm (tdBN), normalising pre-activations to `N(0, (αV_th)²)` to "balance pre-synaptic
  input and threshold to maintain a reasonable firing rate."

**What this costs us:** `calibrate_gain` cannot be presented as a contribution. It is a
re-derivation of established practice for the STDP setting.

**What survives:** the framing inverts usefully. The `[64,32,64]` rpeak collapse
(`phase3_findings.md` §5.6.3) and the Finding 3 saturate/silence failure modes become a
*demonstration of what happens when this known and solved problem is left unsolved in an
unsupervised STDP pipeline* — where, unlike in ANN-to-SNN conversion, there is no reference
activation distribution to normalise against. That is a legitimate and citable observation.
The honest sentence is "we apply established threshold-balancing practice (Diehl 2015,
Rueckauer 2017) to the unsupervised STDP setting, where it has not previously been required,"
not "we introduce gain calibration."

---

## Draft positioning sentence

> Rate-coded spiking pipelines are known to lose information relative to temporal and direct
> coding (Castagnetti 2023), and population codes are known to saturate in the presence of
> shared input noise (Moreno-Bote 2014, Kanitscheider 2015); we connect these two results by
> measuring, in a fully reproducible unsupervised-STDP reconstruction pipeline, a hard output-SNR
> ceiling that is invariant to output width, decoder time constant, training duration, encoder,
> decoder, and signal class — and show that it is set at the encoder, not by the learning rule
> or the network capacity.

Note this sentence promises the M2 result ("set at the encoder"). It is not yet earned.

---

## A. Information limits in population codes — the mechanism

**A1. Moreno-Bote, R., Beck, J., Kanitscheider, I., Pitkow, X., Latham, P., Pouget, A. (2014).
Information-limiting correlations. *Nature Neuroscience* 17, 1410–1417.** **[read]**
The load-bearing theory citation. Only differential correlations — those proportional to the
product of tuning-curve derivatives — can make Fisher information saturate with population size;
decorrelation does *not* imply more information. Cite for: the `I_N = I₀/(1 + ε·I₀)` saturation
form and the 1/ε ceiling; the "finite information in the world ⇒ saturation" argument; and the
shared-input intuition that explains our tiled-input result.

**A2. Kanitscheider, I., Coen-Cagli, R., Pouget, A. (2015). Origin of information-limiting noise
correlations. *PNAS* 112(50), E6973–E6982.** **[abs]** (publisher 403s; abstract verified)
Locates the origin of information-limiting correlations in noise injected at the sensory
periphery and propagated forward, with overlap in linearised receptive fields setting correlation
magnitude. Cite for: our noise is injected at the input and propagated, which is precisely this
regime. **Needs opening before citing in detail.**

**A3. "Information-Limiting Correlations in Large Neural Populations." *Journal of Neuroscience*
40(8), 1668 (2020). PMID 31941667.** **[meta]**
Extends the saturation analysis to large populations; the search-level summary states information
grows linearly or superlinearly at zero/negative correlation but saturates to a finite value for
positive correlation, and that saturation can disappear under heterogeneous tuning. **Author list
not verified — open before citing.** The heterogeneous-tuning escape clause is directly relevant
to R5's non-tiled delay-line embedding experiment.

**A4. "Information-Limiting Correlations in Neural Populations: The Devil Is in the Details."
PMC7548692.** **[read]**
Read in full; gives the phenomenon qualitatively ("correlated noise thus places an upper bound on
the stimulus information") but **no closed-form saturation condition and no DPI framing**. Useful
as a caution that the details matter; not a source for equations. Author list not verified.

**A5. Zohary, E., Shadlen, M.N., Newsome, W.T. (1994).** Correlated neuronal discharge rate and
its implications for psychophysical performance. **[meta]** — cited by A1/A4 as the origin of the
"correlations limit information" line. Historical anchor only.

---

## B. Rate coding capacity and coding-scheme comparison

**B1. Guo, W., Fouda, M.E., Eltawil, A.M., Salama, K.N. (2021). Neural Coding in Spiking Neural
Networks: A Comparative Study for Robust Neuromorphic Systems. *Frontiers in Neuroscience* 15,
638474.** **[read]**
Head-to-head on rate / TTFS / phase / burst coding. Cite for: TTFS needs 4×–7.5× lower latency and
3.5×–6.5× fewer synaptic operations than rate coding; MNIST accuracy rate 87.46% vs TTFS 88.57%,
phase 88.18%, burst 88.39%; phase coding most noise-robust, TTFS most noise-sensitive.

> **Correction to an earlier note.** The widely repeated claim that rate coding "needs 2⁸ = 256
> timesteps to represent 8 bits" surfaced in search summaries attributed to this paper. **It is
> not in it.** I checked. Do not cite B1 for it. If we want that claim, its real source must be
> found first — or we derive the bound ourselves in M3, which is the better option anyway.

**B2. Susemihl, A., Meir, R., Opper, M. (2012). Dynamic State Estimation Based on Poisson Spike
Trains — Towards a Theory of Optimal Encoding. arXiv:1209.5559.** **[read]**
Optimal filtering of a continuous state from Poisson spike trains. Contains the result that
"the distribution of possible errors diverges around the upper bound for the error", implying a
**threshold firing rate below which the input cannot be tracked at all**. Cite for: the existence
of a rate floor for trackability, and as the closest existing analytical treatment for M3. Its
framing is optimal-estimator; ours is a fixed suboptimal decoder, which is a real gap between the
two and worth stating explicitly.

**B3. Chattopadhyay, A., Banerjee, A. (2024). Robust online reconstruction of continuous-time
signals from a lean spike train ensemble code. arXiv:2408.05950.** **[read]**
Deterministic threshold-based (not Poisson) encoding of continuous signals with online
reconstruction. Cite as: the contrasting design point — deterministic encoders reconstruct
continuous signals well, which sharpens the question of why the stochastic Poisson path does not.

**B4. Auge, D., et al. (2021). A Survey of Encoding Techniques for Signal Processing in Spiking
Neural Networks. *Neural Processing Letters*.** **[abs]**
Standard survey reference for encoder taxonomy. Contains the useful framing that "when encoding an
analog number, there is a remaining reconstruction error due to the discretised number of spikes
during a given time interval." Author list beyond first author not verified.

**B5. "Temporal code versus rate code for binary Information Sources." arXiv:1603.02798 /
*Neurocomputing*.** **[meta]** — information-theoretic comparison, potentially useful for M3.

**B6. "Rate coding versus temporal order coding: a theoretical approach."** **[meta]** —
older theoretical treatment; open if M3 needs additional grounding.

---

## C. Gain and threshold normalisation in SNNs

**C1. Diehl, P.U., Neil, D., Binas, J., Cook, M., Liu, S.-C., Pfeiffer, M. (2015).
Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing.
*IJCNN 2015*. doi:10.1109/IJCNN.2015.7280696.** **[abs]**
The canonical prior art for our `calibrate_gain`. Weight normalisation and threshold balancing are
equivalent formulations for regulating firing rates. Cite for: this problem is known and solved in
the conversion setting.

**C2. Rueckauer, B., Lungu, I.-A., Hu, Y., Pfeiffer, M., Liu, S.-C. (2017). Conversion of
Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification.
*Frontiers in Neuroscience* 11, 682.** **[abs]**
Robust percentile normalisation, λ set to the p-th percentile of layer activity, p ∈ [99.0,
99.999]; 12.18% CIFAR-10 error at the 99.9th percentile. Cite for: the mature form of
rate-calibration practice.

**C3. Zheng, H., et al. (2021). Going Deeper With Directly-Trained Larger Spiking Neural Networks.
*AAAI* 35(12), 11062–11070.** **[abs]**
tdBN — normalises pre-activations to `N(0, (αV_th)²)` across both temporal and spatial dimensions
to hold firing rates reasonable. Cite for: normalisation is standard in *directly trained* SNNs
too, not just converted ones. Full author list not verified.

**C4. Membrane Potential Batch Normalization for Spiking Neural Networks. arXiv:2308.08359.**
**[meta]** — normalises membrane potential post-charging. Completeness citation.

**C5. Revisiting Batch Normalization for Training Low-Latency Deep SNNs From Scratch. *Frontiers
in Neuroscience* 15, 773954 (2021).** **[meta]** — completeness citation.

---

## D. SNN reconstruction, denoising, autoencoders — the competitive set

**D1. Castagnetti, A., Pegatoquet, A., Miramond, B. (2023). SPIDEN: deep Spiking Neural Networks
for efficient image denoising. *Frontiers in Neuroscience* 17, 1224457.** **[read]**
**The most important competitive citation.** Supervised, surrogate-gradient, residual (predicts
noise). Uses direct encoding. Their rate-coded ablation (DnSNN-Rate) is "almost 8 dB lower
compared to the other SNNs" because "due to the stochastic nature of the generation process, a
significant amount of information is lost." Results: BSD68 σ=15, 31.593 dB SNN vs 31.73 dB ANN;
Set12 σ=25, 30.186 vs 30.436 — gap consistently under 0.4 dB. They also formalise spike
quantisation noise (SQNR) as a fundamental limit.
**Implication:** the direction of our claim is already published. Our contribution must be the
*characterisation and mechanism*, not the existence of the effect. Also: a well-designed SNN
denoiser gets within 0.4 dB of its ANN — so "SNNs can't denoise" is false and we must not say it.

**D2. Walters, B., et al. (2024). Efficient sparse spiking auto-encoder for reconstruction,
denoising and classification. *Neuromorphic Computing and Engineering* 4(3), 034005.** **[read]**
Closest prior art on the *unsupervised* axis: STDP plus a novel error-modulated STDP variant,
reconstruction and denoising, no labels. But **TTFS encoding, not rate** — explicitly contrasted
with "Poisson rate schemes that require many more spikes" — and images only (MNIST,
Fashion-MNIST). Reports its own degeneracy mode: ~200 test images (2%) produced *no* reconstruction
due to sparse hidden activity. Cite for: STDP-only reconstruction is established; the rate-coded
and time-series cases are the gap.

**D3. Spiking Autoencoders With Temporal Coding. *Frontiers in Neuroscience* (2021).** **[meta]**
— temporal-coding autoencoder; supports the pattern that reconstruction work chooses temporal
coding.

**D4. Spiking neural networks for nonlinear regression of complex transient signals on sustainable
neuromorphic processors. *npj Unconventional Computing* (2024).** **[meta]**
Could not fetch (Nature IDP redirect). SNN *regression* on transient signals — potentially very
close to our task. **Highest-priority remaining fetch.**

---

## E. SNNs for biomedical signals and ECG

**E1. Ren, Y., Zuo, H., Cai, Y., Wang, S., Ran, G., Lai, D. (2026). DCPA-SNN: Direct-Coding
Physics-Aware Spiking Neural Network — A Framework for Wearable ECG Denoising Under Dynamic-Noise
Conditions. *Sensors* 26(15), 4695.** **[read]**
**The direct competitor, and very recent (July 2026).** Supervised, direct coding, clinically
weighted reconstruction loss. Average SNR improvement **+6.80 ± 3.09 dB** over −6 to 4 dB input;
R-peak detection 95.72% vs 90.71% noisy baseline. They claim the gap we were going to claim:
"to the best of our knowledge, this work explores an end-to-end SNN-based framework specifically
designed for wearable ECG denoising."
**Two things follow.** (i) We cannot claim novelty on "end-to-end SNN for ECG denoising."
(ii) Their per-regime split — **8.82 dB improvement at −6 to −1 dB, 4.38 dB at 0 to 4 dB** — is the
*same monotone decrease with input SNR* that Phase 3 found. Independent corroboration of our
regime-dependence finding, from a supervised system. That is a genuinely useful result for us.

**E2. Choi, S.H. (2024). Spiking neural networks for biomedical signal analysis. *Biomedical
Engineering Letters* 14(5), 955–966.** **[read]**
Reviews SNN work on ECG/EEG/EMG. Confirms: "Most SNNs research based on ECG signals has focused on
arrhythmia detection" — all seven ECG studies reviewed are classification. **No denoising or
regression discussed at all.** Cite for: the classification/reconstruction imbalance in the
biomedical SNN literature, stated by a review rather than asserted by us.

**E3. Exploring the potential of spiking neural networks in biomedical applications: advantages,
limitations, and future perspectives. *Biomedical Engineering Letters* (2024). PMC11362408.**
**[read]**
Explicit limitations: EEG accuracy "lower compared to other neural networks"; ECG "improving
accuracy remains a future challenge"; EMG "not consistently higher when compared to LSTM." Does
*not* discuss rate coding as the cause. Cite for: SNN underperformance on biomedical signals is
acknowledged in review literature but not mechanistically explained — which is our opening.

**E4. Neuromorphic implementation of ECG anomaly detection using delay chains. arXiv:2209.01266.**
**[meta]** — classification, not denoising. Context only.

**E5. A neuromorphic approach to early arrhythmia detection. *Scientific Reports* (2025).**
**[meta]** — classification. Context only.

---

## F. Unsupervised STDP

**F1. Diehl, P.U., Cook, M. (2015). Unsupervised learning of digit recognition using
spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience* 9, 99.** **[abs]**
The canonical "STDP learns structure without labels" reference; up to 95% MNIST unsupervised;
mechanisms are exponential synapses with STDP, lateral inhibition, adaptive threshold; the authors
liken it to k-means / vector quantisation.
> **Citation correction:** an early search returned this as *IEEE TNNLS* 26(12), 2999–3008.
> **That is wrong.** Verified as Front. Comput. Neurosci. 9:99. Use the verified form.

**F2. An unsupervised STDP-based spiking neural network inspired by biologically plausible learning
rules and connections. *Neural Networks* (2023); arXiv:2207.02727.** **[meta]** — STB-STDP,
current state of the art for unsupervised SNNs; scales to CIFAR-10. Cite as the modern reference
point for what unsupervised STDP can achieve.

**F3. A Heterogeneous Spiking Neural Network for Unsupervised Learning of Spatiotemporal Patterns.
PMC7841292.** **[meta]** — STDP on spatiotemporal patterns; closest unsupervised analogue to
time-series work.

---

## G. Classical ECG denoising baselines — for the motivating section only

**G1. Denoising ECG by Adaptive Filter with Empirical Mode Decomposition. arXiv:2108.08376.**
**[meta]** — parallel EMD + adaptive filter; reports best SNR improvement on MIT-BIH across BW,
PLI, EM and MA noise. Use for Phase 4's classical baseline set and for the standard metric
vocabulary (MSE, RMSE, PRD, SNR_imp).

**G2. Empirical Wavelet Transform Based ECG Signal Filtering Method. *Journal of Electrical and
Computer Engineering* (2024).** **[meta]** — wavelet baseline reference.

**G3. Does the Data Processing Inequality Reflect Practice? On the Utility of Low-Level Tasks.
*ICLR 2026*; arXiv:2512.21315.** **[meta]**
Directly relevant to our framing: the DPI says denoising cannot increase information, yet
denoising demonstrably helps at finite sample sizes. If we invoke DPI at all, this is the paper
that says why the naive invocation is too strong. **Should be read before R6.**

---

## What R0 changes about the roadmap

Four concrete consequences. All four are in the project's favour except the first.

**1. The `N_in` capacity result is weaker than the roadmap claims, and must be reframed.**
Finding 4 reports +0.78 dB over a 32× `N_in` range against a "DPI predicts +15 dB" expectation.
That expectation was wrong: with `np.tile`, all input channels share one noise realisation, so
information about the clean signal does not grow with `N_in` at all. A reviewer who knows
Moreno-Bote will make this objection immediately, and they will be right. **Do not publish
Finding 4 in its current framing.**

**2. But this hands us the decisive experiment, and it is cheap.**
Replace the tiled input with genuinely independent draws — either N independent noise realisations
of the same clean signal, or the delay-line embedding already planned in R5 — and re-run the
`N_in` sweep under per-layer rate control.
- If the ceiling **lifts** → we have confirmed the mechanism, quantified where a practical
  rate-coded pipeline sits relative to the population-coding bound, and can state the design rule
  ("tiling a scalar across an input population buys you nothing"). Solid, useful, publishable.
- If the ceiling **holds** → information-limiting correlations do *not* explain it, something
  downstream binds, and that is a substantially more interesting paper.
Either outcome is a result. This should become the first experiment of R4, ahead of everything else.

**3. `calibrate_gain` is engineering, not contribution.** Ship it — every ceiling claim depends on
it — but cite Diehl 2015 / Rueckauer 2017 / Zheng 2021 and describe it as applying known practice
to the STDP setting. Recover the value by reframing the `[64,32,64]` collapse as a demonstration
of what unnormalised rate-coded pipelines do.

**4. Novelty is intact but narrower than hoped, and the target venue may shift.**
- Taken: end-to-end SNN ECG denoising (E1, 2026); rate coding is lossy for denoising (D1, ~8 dB);
  STDP-only reconstruction and denoising (D2); information saturation under shared input (A1).
- Open: a systematic characterisation of the ceiling's invariance structure across width, τ,
  training, encoder, decoder and signal class; the mechanism localised to a pipeline stage; and
  the connection between the population-coding theory and practical rate-coded SNN engineering,
  which nothing found makes.
That is a real contribution and it is defensible. It is a **characterisation-and-mechanism paper**,
not a discovery paper, and it should be written and pitched as one.

---

## Remaining R0 work

- [ ] Open D4 (npj regression) — highest priority; potentially the closest task match.
- [ ] Open A2 (Kanitscheider PNAS) via an institutional or preprint route; publisher 403s.
- [ ] Open A3 and verify its author list.
- [ ] Verify full author lists for B4, C3, A3, A4.
- [ ] Find the real source of the "2ⁿ timesteps for n bits" claim, or drop it and derive in M3.
- [ ] Read G3 before writing any DPI argument in R6.
- [ ] Search once more specifically for rate-coded SNN *time-series regression* — the searches so
      far were image-dominated, and D4 suggests that literature exists.
