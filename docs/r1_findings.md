# R1 — Measurement validity: the clinical metric was measuring nothing

**Status:** resolved. `metrics/distortion.py` and `metrics/qrs.py` are in the
library and tested; `rpeak_integrity` is deprecated in place.

> **Read §4 and §5 in their corrected form.** The first version of this document
> attributed a regime advantage to "the passive network" using a scaled-init
> network that Phase 3 never ran, and described calibration's QRS collapse as
> "the same trade-off Phase 3 found for STDP". Both are corrected below; the
> saturation result in §2 is unaffected.
> **Correction (R3, `docs/r3_findings.md` §1.6) — input-SNR labels, and what "doing nothing"
> means.** **(1)** Every MIT-BIH input SNR in this document is *nominal*. `make_pair` adds the 50 Hz
> flicker at 0.5 amplitude first and scales the Gaussian noise to the power of the flickered
> signal, so on unit-variance ECG the flicker alone holds the input below +9 dB. Measured against
> the clean signal (16 records × 3 seeds), nominal −5 / 0 / 5 / 10 / 20 dB are actually
> **−5.6 / −0.9 / +3.2 / +6.3 / +8.7 dB**. The 25 dB span in §2 is therefore a 14.3 dB span of true input SNR; `rpeak_integrity` still moves by 0.000 across it. No direction reported here changes; the
> positions on the SNR axis do. **(2)** The "unprocessed" or "doing nothing" QRS baseline runs
> neurokit's peak detector on the raw input, which never removes the flicker. After neurokit's
> own standard `ecg_clean` (0.5 Hz high-pass plus powerline filter) the same detector scores
> **median F1 0.92 at nominal −5 dB and 1.00 at every other level**; the best median of any
> NeuroFlux configuration in R2's sweep is 0.71 (10 epochs, nominal 20 dB), though single records
> reach 0.98. Network-versus-identity QRS results are comparisons against an unfiltered detector,
> not against conventional ECG preprocessing.

**Source data:** `results/r1_metrics/saturation.csv` — 16 MIT-BIH records x 3
seeds x 5 input SNRs x 4 conditions, 768 rows.
**Reproduce:** `uv run python scripts/r1_metrics/saturation.py`, or
`--analyse-only` to reprint every table from the committed CSV.

---

## 1. The defect

`rpeak_integrity(clean, reconstructed, fs)` detects peaks in both signals with
`scipy.signal.find_peaks` and matches them within 50 ms. Its prominence floor
is `0.5 * std` of **each signal independently** (`rpeak.py:54-62`).

That single line is the whole problem. The threshold is a fraction of the
signal's own standard deviation, so adding noise raises the bar in exact
proportion to the noise added. The detector is handed a harder signal and a
correspondingly more permissive threshold, and the two cancel.

The reference is also circular. The score compares detections against
detections — two runs of the same algorithm — and never against ground truth.
A detector that is wrong in the same way on both inputs scores perfectly.

## 2. How large the defect is

Score the **unprocessed noisy signal**. Nothing is done to it; the only thing
that changes between columns is how much noise it carries.

| metric | at −5 dB in | at 20 dB in | span |
|---|---|---|---|
| **`rpeak_integrity`** | **1.000** | **1.000** | **0.000** |
| `qrs_f1` (annotation-referenced) | 0.046 | 0.683 | 0.638 |
| `prd1` (%) | 191.4 | 36.8 | 154.6 |
| `max_cross_correlation` | 0.461 | 0.938 | 0.477 |
| affine-corrected SNR (dB) | 1.04 | 9.23 | 8.19 |

Across a **25 dB** change in nominal input noise (14.3 dB of true input SNR — see the
correction above), `rpeak_integrity` moves by **zero**,
to three decimals, with a standard deviation of 0.000 across 16 records. It is
a constant function on this domain. Every other metric responds.

This is not a metric that performs poorly. It is a metric that is not
connected to its input.

## 3. What it means for Phase 3 §5.6.2

Phase 3's most-cited diagnostic is that the passive, untrained network
"preserves R-peaks perfectly", `rpeak_integrity = 1.000` at every input SNR.
That number is the artifact, not a property of the network. The identity
baseline — doing nothing at all — scores exactly the same 1.000.

Measured on the same runs, `rpeak_integrity` cannot distinguish:

- a clean MIT-BIH record (1.000),
- the same record at −5 dB SNR with 50 Hz flicker (1.000),
- a reconstruction whose affine-corrected SNR is 0.3 dB (0.996–1.000).

The claim "the passive architecture preserves clinical landmarks" is therefore
**unsupported by the evidence given for it**. Section 4 measures it properly,
and it turns out to be largely false.

## 4. What the replacement finds

> **Correction (second pass).** The first version of this section reported that
> "the passive network" beats the identity baseline at −5 and 0 dB and has no
> effect above. That arm was built with `init_scheme="scaled"`
> (`saturation.py:119`), but the Phase 3 runner — and so every Phase 3 number —
> uses the `Network.feedforward` default, `"uniform"`. The two passive networks
> behave very differently. The table below is re-measured on the network Phase 3
> actually ran, at 10 seeds; the scaled-init result is kept, labelled, because it
> turns out to be informative (§4.2).

`metrics/qrs.py` scores `neurokit2.ecg_peaks` detections against the
cardiologist annotations in each record's `.atr` file, at the ANSI/AAMI EC57
tolerance of ±150 ms, reporting sensitivity, PPV, F1 and RMS timing jitter.
Beats within one second of either window edge are excluded, because a detector
needs context to declare its first beat and its warm-up is not a clinical miss.
On the clean signal the detector reaches mean F1 **0.988** over 48 runs, so
that is the ceiling any processed score is read against.

### 4.1 The Phase 3 networks against doing nothing

> **Detector check (R5, `docs/r5_findings.md` §2.2–§2.3).** These comparisons use `neurokit`'s
> detector on unfiltered signals. After `ecg_clean`, the passive network's F1 at ≥ 5 dB rises to
> 0.57–0.78. It still falls short of the cleaned input's 1.00, but "destroys" overstates the harm.

Source: `results/r2_reanalysis/qrs_rescore.db` (16 records × 5 input SNRs ×
{0, 10, 50} epochs × 10 seeds, `configs/r2/qrs_rescore.yaml`) and
`results/r2_reanalysis/identity_comparisons.csv`, from
`scripts/r2_reanalysis/identity_baseline.py`, which re-scores the unprocessed
input on exactly the windows and noise draws the networks saw. Network minus
identity, median ΔF1 across records, Wilcoxon on record, Holm across the five
SNRs, TOST margin ±0.05.

| input SNR | passive (0 ep) | 10 epochs | 50 epochs |
|---|---|---|---|
| −5 dB | **+0.123** (p=0.019) | **+0.406** (p=0.0002) | **+0.363** (p=0.0002) |
| 0 dB | −0.061 (inconclusive) | **+0.364** (p=0.0006) | **+0.256** (p=0.005) |
| 5 dB | **−0.553** (p=0.022) | +0.101 (inconclusive) | +0.011 (inconclusive) |
| 10 dB | **−0.754** (p=0.011) | −0.011 (inconclusive) | −0.090 (inconclusive) |
| 20 dB | **−0.756** (p=0.013) | −0.048 (inconclusive) | −0.091 (inconclusive) |

Three statements follow, and they overturn the Phase 3 story rather than
refining it.

1. **The Phase 3 passive network destroys QRS recoverability above 5 dB.** At
   10 and 20 dB input the unprocessed signal lets the detector find beats; the
   passive network's output does not, by three quarters of the F1 scale. It helps
   only at −5 dB, and by much less than the trained networks do.
2. **The low-SNR advantage belongs to the trained network.** Ten epochs of STDP
   recover beats that are otherwise undetectable at −5 and 0 dB, and are
   indistinguishable from doing nothing above 5 dB — the same crossover Phase 3
   reports for ΔSNR, now on an independent metric with an independent reference.
3. **STDP repairs landmarks rather than damaging them.** See `docs/r2_findings.md`
   §5: on the same runs, `rpeak_integrity` reports the effect of training with
   the wrong sign.

"Inconclusive" at 5–20 dB for the trained networks means neither a difference
nor equivalence within ±0.05 was established; the confidence intervals span
roughly ±0.1–0.25, so the honest reading is "no detectable effect", not "equal".

### 4.2 The scaled-init passive network (the original R1 arm)

Same metric, 3 seeds, `init_scheme="scaled"`, no training:

| input SNR | n | median ΔF1 vs identity | p (Holm) |
|---|---|---|---|
| −5 dB | 16 | +0.366 | 0.0003 |
| 0 dB | 16 | +0.224 | 0.0305 |
| 5 / 10 / 20 dB | 16 | −0.020 / −0.130 / −0.064 | 1.00 |

An untrained network with weights drawn on `[0, 0.3]` instead of `[0, 1]`
behaves like the *trained* uniform network, not like the untrained one. §5 says
why.

## 5. What actually preserves landmarks: a sparse output

> **Detector check (R5, `docs/r5_findings.md` §2.1, §2.4).** A sparse output helps the unfiltered
> detector. After `ecg_clean` a dense 40 Hz output is as detectable as a 1 Hz one (F1 0.81 each),
> so what a dense output loses here is hidden from an unfiltered detector, not destroyed.

Two observations had to be reconciled. M1b's per-layer gain calibration, which
every mechanism experiment from M2 on uses, collapses QRS F1 to about zero at
every input SNR (scaled-init network, 3 seeds):

| input SNR | scaled passive F1 | scaled passive + calibration F1 |
|---|---|---|
| −5 dB | 0.423 | 0.000 |
| 0 dB | 0.464 | 0.000 |
| 5 dB | 0.536 | 0.003 |
| 10 dB | 0.599 | 0.019 |
| 20 dB | 0.638 | 0.022 |

And initial weight scale, which calibration exists to neutralise, decides
whether the untrained network works at all (§4). A direct measurement on four
records at 10 dB input, seed 0, reads both as one effect:

| network | layer-1 mean weight | output rate | QRS F1 |
|---|---|---|---|
| uniform init, 0 epochs | 0.53 | 55–85 Hz | 0.00–0.17 |
| scaled init, 0 epochs | 0.16 | 0.5–5 Hz | 0.55–0.88 |
| uniform init, 10 epochs | 0.05–0.09 | 0.2–1.2 Hz | 0.00–1.00 |
| any init, calibrated to 40 Hz | — | 40 Hz | ≈ 0 |

A QRS complex is recoverable from the decoded output when the output layer
fires *sparsely* — rarely, and mostly on the complex — so the decoded trace is a
baseline with bumps where the beats are. When it fires densely, whether because
weights are large (uniform init), because calibration forces 40 Hz, the bursts
that mark a beat are lost in continuous activity. STDP from uniform init
depresses layer-1 weights by roughly an order of magnitude, which moves the
network into the sparse regime; that is the most direct candidate for why
training helps F1.

**This is the real trade-off, and it is an operating-point trade-off, not an
STDP trade-off.** A dense, rate-calibrated output is what the cascade
measurements need and what reconstruction SNR rewards; a sparse output is what
landmark detection needs. Phase 3's "STDP buys ΔSNR at the cost of R-peaks" was
the saturated metric's rendering of this, with the sign of the R-peak half
reversed.

Stated limits: the mechanism table is four records, one seed, one input SNR,
and the sparse regime is not sufficient on its own — record 214 after training
fires at 0.2 Hz and scores F1 0.000, too silent to mark every beat. It is a
hypothesis with direct supporting evidence, not an established result. Mapping
F1 over output rate is R3 C3's second axis, and it is where this gets settled.

## 6. The distortion metrics

`metrics/distortion.py` adds the vocabulary an ECG reader expects:

- `prd` and `prd1` — percentage root-mean-square difference. `prd` divides by
  raw signal energy and so improves when a DC offset is added, with the
  reconstruction error unchanged; `prd1` divides by energy about the mean and
  is comparable across records. **Report `prd1`.** The two are not
  interchangeable and the literature often does not say which it quotes.
- `rmse` — scale-dependent, meaningful here only because Phase 3 normalises
  every record to unit variance.
- `max_cross_correlation` / `cross_correlation_peak` — morphology under a
  timing tolerance, with the lag returned. A decoder with a fixed group delay
  reconstructs the waveform faithfully and scores badly on every
  sample-aligned metric; this separates "wrong shape" from "right shape,
  shifted".

`prd1` and the affine-corrected SNR are the same statement in different units
whenever the residual is centred — pinned as a test — so reporting both costs
nothing and buys the field's vocabulary.

The passive network's `prd1` (uncalibrated arm of `saturation.py`, scaled init) is **274–638%** across the SNR range: the residual
carries two to six times the energy of the signal. Reported in the field's own
terms, that is the honest scale of the reconstruction problem, and it is
consistent with the 1.5 dB ceiling M3 derived.

## 7. Consequences

1. **`rpeak_integrity` is deprecated in place**, with the reason and the
   pointer in its module and function docstrings. It is not deleted: Phase 3's
   published numbers are expressed in it, and removing it would make them
   unreadable rather than corrected.

2. **Phase 3 §5.6.2's headline is retracted, not restated.** "The passive
   network preserves R-peaks; STDP damages them" is inverted on a working metric:
   the Phase 3 passive network destroys QRS recoverability above 5 dB, and STDP
   restores it. Correction blocks are in `docs/phase3_findings.md` (banner and
   §5.6.2) and `docs/r2_findings.md` §5.

3. **Every future clinical claim is annotation-referenced.** `qrs_score`
   requires reference beats; there is no code path that scores detections
   against detections.

4. **There is no synthetic counterpart to `load_reference_beats`, deliberately.**
   `generate_ecg` builds its waveform from a known beat train, so scoring a
   detector against those positions measures the generator. Ground truth here
   means a human marked it.

5. **R3's operating-region work now has a second axis.** §5 says landmark
   recoverability depends on how sparsely the output fires, and rate
   calibration forces the opposite. The (output rate, ΔSNR, QRS F1) surface is
   the result to draw, not a nuisance to average away.

## 8. What R1 did not settle

- **The detector is a dependency of the result.** Every number above uses
  `neurokit2`'s default method. `detect_rpeaks` passes `method` through so
  `pantompkins1985` and `hamilton2002` can be run as a robustness check; that
  check has not been run. It belongs in R5's generality grid.
- **The clean-signal ceiling is uneven across records.** Twelve of the 16
  reach F1 = 1.000 on the unprocessed clean window; 232 reaches 0.889, 113
  0.961, 108 0.978 and 231 0.980. Those four cap what any processing can score
  on them. They are kept rather than dropped — excluding the hard records to
  raise the mean is the failure mode this stage exists to prevent — but a
  per-record table has to show the ceiling next to the score.
- **Timing jitter is dominated by three records.** On the clean signal, most
  records sit at 1.5–5 ms RMS; record 108 is at 55.3 ms, 207 at 26.1 ms and
  200 at 20.8 ms, where the annotation and the detector disagree about where
  within the complex the R-peak sits. Pooled jitter figures inherit that and
  should be reported per record or as a median.
- **Three seeds, not ten.** The paired tests are over 16 records, which is the
  unit of analysis, so the seed count affects precision rather than validity —
  but the roadmap's ≥10-seed commitment is still outstanding here.
