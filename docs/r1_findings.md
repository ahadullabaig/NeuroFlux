# R1 — Measurement validity: the clinical metric was measuring nothing

**Status:** resolved. `metrics/distortion.py` and `metrics/qrs.py` are in the
library and tested; `rpeak_integrity` is deprecated in place.
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

Across a **25 dB** change in input noise, `rpeak_integrity` moves by **zero**,
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
**unsupported by the evidence given for it**. Section 4 shows it is not thereby
false — but it needed a different measurement to establish, and it comes with a
regime restriction Phase 3 did not state.

## 4. What the replacement finds

`metrics/qrs.py` scores `neurokit2.ecg_peaks` detections against the
cardiologist annotations in each record's `.atr` file, at the ANSI/AAMI EC57
tolerance of ±150 ms, reporting sensitivity, PPV, F1 and RMS timing jitter.
Beats within one second of either window edge are excluded, because a detector
needs context to declare its first beat and its warm-up is not a clinical miss.
On the clean signal the detector reaches mean F1 **0.988** over 48 runs, so
that is the ceiling any processed score is read against.

Passive network minus identity baseline, paired on record, Wilcoxon
signed-rank, Holm-Bonferroni across the five input SNRs:

| input SNR | n | median ΔF1 | p | p (Holm) | verdict |
|---|---|---|---|---|---|
| −5 dB | 16 | **+0.366** | 0.0001 | **0.0003** | advantage |
| 0 dB | 16 | **+0.224** | 0.0076 | **0.0305** | advantage |
| 5 dB | 16 | −0.020 | 0.90 | 1.00 | no effect |
| 10 dB | 16 | −0.130 | 0.56 | 1.00 | no effect |
| 20 dB | 16 | −0.064 | 0.56 | 1.00 | no effect |

**Phase 3's regime claim survives the metric fix, and is now supported by a
clinical measurement with human ground truth.** Phase 3 found the SNN improves
ΔSNR at input SNR ≤ 0 dB and degrades it at ≥ 5 dB. Annotation-referenced QRS
F1 reproduces that boundary exactly, on an independent metric with an
independent reference. Two different measurements, the same crossover.

What changes is the *scope* of the claim. It is not "R-peaks are preserved
perfectly"; it is "below 0 dB the network recovers beats that are otherwise
undetectable, and above 5 dB it neither helps nor hurts."

## 5. Gain calibration trades morphology for rate fidelity

The runs include a fourth condition: the same passive network with M1b's
per-layer gain calibration, which every mechanism experiment from M2 onward
uses. It is catastrophic for QRS:

| input SNR | passive F1 | passive + calibration F1 |
|---|---|---|
| −5 dB | 0.423 | 0.000 |
| 0 dB | 0.464 | 0.000 |
| 5 dB | 0.536 | 0.003 |
| 10 dB | 0.599 | 0.019 |
| 20 dB | 0.638 | 0.022 |

Calibration drives every layer to a 40 Hz mean rate. That is the right thing to
do when the quantity of interest is a rate code — it is why M1b introduced it,
and it is what makes the cascade measurements in M2/M3 comparable across
configurations. It is the wrong thing to do when the quantity of interest is
the location of a transient: flattening the rate distribution removes exactly
the bursts that mark a QRS complex.

This is the same trade-off Phase 3 §5.6.2 found for STDP (ΔSNR up, R-peaks
down), arrived at from a different direction, and it is now visible in a
clinical metric rather than a saturated one. **Rate fidelity and landmark
preservation are separate objectives in this architecture, and the operating
point that optimises one degrades the other.**

Caveat, stated rather than buried: the calibration target used here (40 Hz,
`tau_decode` 20 ms) was chosen for the cascade measurements, not for QRS. The
finding is that *this* operating point trades morphology away, not that no
calibrated operating point can preserve it. Locating one is R3's job.

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

The passive network's `prd1` is **274–638%** across the SNR range: the residual
carries two to six times the energy of the signal. Reported in the field's own
terms, that is the honest scale of the reconstruction problem, and it is
consistent with the 1.5 dB ceiling M3 derived.

## 7. Consequences

1. **`rpeak_integrity` is deprecated in place**, with the reason and the
   pointer in its module and function docstrings. It is not deleted: Phase 3's
   published numbers are expressed in it, and removing it would make them
   unreadable rather than corrected.

2. **Phase 3 §5.6.2's headline needs restating**, not retracting. A correction
   block is appended to `docs/phase3_findings.md` §5.6.2.

3. **Every future clinical claim is annotation-referenced.** `qrs_score`
   requires reference beats; there is no code path that scores detections
   against detections.

4. **There is no synthetic counterpart to `load_reference_beats`, deliberately.**
   `generate_ecg` builds its waveform from a known beat train, so scoring a
   detector against those positions measures the generator. Ground truth here
   means a human marked it.

5. **R3's operating-region work now has a second axis.** The (ΔSNR, F1)
   trade-off surface is not a nuisance to be averaged away; §5 says the
   architecture has two separate optima and the paper should show both.

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
