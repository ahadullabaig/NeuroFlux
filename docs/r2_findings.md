# R2 — statistical validity, and what it did to the Phase 3 claims

**Status:** complete. The defect is fixed, the protocol is implemented and tested, and the
Phase 3 claims have been re-analysed at the correct unit of analysis.
**Data:** `results/phase3_r2/`
**Library:** `packages/datasets/src/neuroflux_datasets/analysis.py`, tested in
`tests/datasets/test_statistics.py` (33 tests).
**Reproduce:**
```
uv run python scripts/r2_reanalysis/phase3_reanalysis.py
uv run neuroflux-experiment compare results/phase3_full_sweep.db epochs 10 50 \
    --stratify-by input_snr
```

---

> **Correction (R3, `docs/r3_findings.md` §1.6) — input-SNR labels, and what "doing nothing"
> means.** **(1)** Every MIT-BIH input SNR in §3 and §5 is *nominal*. `make_pair` adds the 50 Hz
> flicker at 0.5 amplitude first and scales the Gaussian noise to the power of the flickered
> signal, so on unit-variance ECG the flicker alone holds the input below +9 dB. Measured against
> the clean signal (16 records × 3 seeds), nominal −5 / 0 / 5 / 10 / 20 dB are actually
> **−5.6 / −0.9 / +3.2 / +6.3 / +8.7 dB**. No direction reported here changes; the
> positions on the SNR axis do. **(2)** The "unprocessed" or "doing nothing" QRS baseline runs
> neurokit's peak detector on the raw input, which never removes the flicker. After neurokit's
> own standard `ecg_clean` (0.5 Hz high-pass plus powerline filter) the same detector scores
> **median F1 0.92 at nominal −5 dB and 1.00 at every other level**; the best median of any
> NeuroFlux configuration in R2's sweep is 0.71 (10 epochs, nominal 20 dB), though single records
> reach 0.98. Network-versus-identity QRS results are comparisons against an unfiltered detector,
> not against conventional ECG preprocessing.


## 1. The defect, and its size

`fetch_raw_runs` selected 8 of the 11 data columns. The three it dropped — `dataset`,
`record_id`, `rpeak_integrity` — were added in Phase 3, so:

- **`rpeak_integrity` was unreachable from the analysis layer entirely.** Phase 3's
  headline clinical metric could not be tested. Every claim about it in
  `phase3_findings.md` was made by reading run output directly, never through the
  significance machinery.
- **All 16 MIT-BIH records collapsed into one group per parameter combination.**
  `_build_phase3_tasks` stamps `signal_type="ecg"` on every record, so with `record_id`
  absent from the key there was nothing left to separate them.

Measured on `results/phase3_full_sweep.db` (2560 runs, 16 records, 5 seeds):

| | before | after |
|---|---:|---:|
| groups | 32 | **512** |
| `n_seeds` per group | 80 | **5** |
| rows with `rpeak_integrity` reachable | 0 | **2560** |

512 = 16 records × 2 flicker × 4 input SNR × 2 epochs × 2 tau, which is the acceptance
criterion. The reported `n = 80` was 5 seeds × 16 patients presented as 80 replicates of
one condition. **Between-patient variance was being counted as seed noise** — textbook
pseudo-replication, and obvious to a reviewer from the *n* alone.

## 2. The protocol

`paired_comparison` implements the roadmap's R2 protocol end to end:

| step | choice |
|---|---|
| unit of analysis | one configuration per record; seeds averaged within unit before testing |
| pairing | units present at both levels of the factor; unpaired units dropped |
| test | Wilcoxon signed-rank on paired differences |
| stratification | any config column, one test per stratum |
| correction | Holm-Bonferroni within the family; `family_size` reported alongside |
| effect size | median paired difference, bootstrap BCa 95% CI, 10 000 resamples, seeded |
| invariance | TOST against a pre-registered margin |

**Four verdicts.** Significance and equivalence are independent tests, so collapsing them
loses a case this project needs:

| | inside margin | outside margin |
|---|---|---|
| **significant** | `trivial` — real but too small to matter | `different` |
| **not significant** | `equivalent` | `inconclusive` |

`trivial` is the verdict Findings 2–5 will mostly earn: "16× the output population changes
the ceiling by 0.16 dB" is a real effect that is also practically nothing, and calling it
either *different* or *equivalent* alone would misrepresent it. `inconclusive` means
neither was established — reported as underpowered, never as invariance. Pinned by
`test_tost_rejects_noisy_data_centred_on_zero`, which uses differences that average near
zero but scatter well past the margin.

Synthetic sweeps are unaffected by the key change: every Phase 2 row shares
`dataset="synthetic"` and `record_id=None`, so the grouping is identical. That is pinned
by `test_synthetic_grouping_is_unchanged`, and the pre-Phase-3 8-column schema still reads
via a `PRAGMA table_info` fallback.

> **Note (R1).** The `rpeak_integrity` comparisons in this section inherit that
> metric's saturation defect, documented in `docs/r1_findings.md` §2: the metric
> does not respond to input noise at all, so a *difference* between two
> conditions measured in it is a difference in how each condition perturbs a
> self-normalising detector, not in clinical landmark preservation. The R2
> protocol applied to it is sound; the quantity it was applied to is not. The
> ΔSNR results in this section are unaffected. Re-running these comparisons on
> `metrics/qrs.py` is outstanding.

## 3. Re-analysis: the Phase 3 claims hold, and get stronger

> **Superseded for R-peaks by §5.** The `rpeak_integrity` rows below are internally correct
> applications of the protocol to a metric that R1 showed does not measure R-peak preservation.
> On QRS F1 the sign of both R-peak results reverses. The ΔSNR rows stand.

### 3.1 Trained vs passive — `phase3_findings.md` §5.6.2

n = 16 records per stratum, paired on (record, input SNR), flicker 50 Hz, tau 20 ms.

| SNR_in | ΔSNR median Δ | 95% CI | rpeak median Δ | 95% CI |
|---:|---:|---|---:|---|
| 0 | **+5.12** | [+3.76, +6.02] | **−0.50** | [−0.68, −0.36] |
| 5 | +4.14 | [+3.50, +5.22] | −0.57 | [−0.71, −0.44] |
| 10 | +3.55 | [+2.87, +4.97] | −0.65 | [−0.75, −0.55] |
| 20 | +3.48 | [+2.58, +4.62] | −0.66 | [−0.77, −0.51] |

All eight comparisons `different` after Holm correction, every CI excluding zero.

**The §5.6.2 trade-off is confirmed and quantified properly for the first time.** STDP buys
+3.5 to +5.1 dB of ΔSNR and costs 0.50 to 0.66 of R-peak integrity. The ΔSNR benefit is
largest at low input SNR and the rpeak damage is largest at high input SNR, so the trade is
worst exactly where the network was already least useful.

### 3.2 More training is worse — and the trade-off is not monotonic

n = 64 per stratum (16 records × 2 flicker × 2 tau), 50 epochs minus 10.

| SNR_in | ΔSNR median Δ | 95% CI | rpeak median Δ | 95% CI |
|---:|---:|---|---:|---|
| 0 | **−1.45** | [−1.77, −1.32] | **+0.20** | [+0.17, +0.25] |
| 5 | −0.88 | [−1.29, −0.72] | +0.27 | [+0.20, +0.30] |
| 10 | −0.43 | [−0.76, −0.20] | +0.20 | [+0.15, +0.25] |
| 20 | −0.30 | [−0.61, −0.08] | +0.16 | [+0.08, +0.22] |

All `different` after correction. This is new: **the ΔSNR/rpeak trade-off reverses between
10 and 50 epochs.** Going 0 → 10 buys ΔSNR and destroys rpeak; going 10 → 50 gives rpeak
back and loses ΔSNR. Training does not move the network along a single trade-off curve, and
**10 epochs — the canonical Phase 3 configuration — is close to the worst point for rpeak.**

This corroborates M2 §6 from a completely different direction. M2 found on synthetic ECG
that STDP saturates by 10 epochs and that further training degrades well-configured
networks; here on real ECG, 50 epochs is worse than 10 on ΔSNR at every input SNR.

## 4. Consequences

**For the manuscript.** Every Phase 3 number that carried `n = 80` must be restated with
`n = 5` per configuration and `n = 16` records as the unit of analysis. The *directions* of
all Phase 3 conclusions survive; only the precision claims change, and they get more
defensible rather than less — 16 independent patients is a stronger basis than 80 correlated
runs, even though the number is smaller.

**For R3.** The invariance claims in Findings 1–5 can now be stated properly. M3 predicts
most of them are invariances of non-binding stages, which TOST can now confirm rather than
merely fail to reject.

**Caveats.**
- The untrained arm exists only at flicker 50 Hz, tau 20 ms, so §3.1 covers 16 of the 64
  units §3.2 uses. A full `epochs=0` sweep would strengthen it.
- TOST margins are declared here (0.5 dB, 0.05 rpeak) and must not be tuned after seeing
  results. They are recorded in `scripts/r2_reanalysis/phase3_reanalysis.py:MARGINS`.
- Wilcoxon at n=16 has limited power; the bootstrap CIs are the more informative column and
  should lead in the manuscript tables.

---

## 5. Re-run on QRS F1: both R-peak conclusions reverse

> **Detector check (R5, `docs/r5_findings.md` §2.2).** Every F1 in this section is `neurokit`'s
> detector on the unfiltered signal. Re-tested on R3 C1's shipped networks (Gaussian noise at true
> SNR) under three detectors, raw and after `ecg_clean`, "10 epochs raise F1 over passive at every
> input" survives none of the six variants. On the unfiltered `neurokit` detector the direction
> replicates (+0.30 to +0.64). After cleaning, the passive network scores 0.33–0.78 instead of
> 0.05–0.16, level with the trained network at 20 dB. "Passive destroys QRS recoverability at
> ≥ 5 dB" becomes "harms it": 0.57–0.78 against the cleaned input's 1.00. The reversal of
> `rpeak_integrity` in §5.1 is unaffected.

**Source:** `results/r2_reanalysis/qrs_rescore.db` — 16 records × 5 input SNRs (−5 to 20 dB)
× {0, 10, 50} epochs × 10 seeds, flicker 50 Hz, τ 20 ms, all in one database from one config
(`configs/r2/qrs_rescore.yaml`). The runner now records annotation-referenced QRS scores next to
the old columns, so every row carries both. Analysis: `scripts/r2_reanalysis/qrs_reanalysis.py`
→ `results/r2_reanalysis/qrs_reanalysis.csv`. Margins fixed before the results were read:
0.5 dB for ΔSNR, 0.05 for F1/Se/PPV and for `rpeak_integrity`, 10 ms for timing jitter.

This also closes §4's first caveat: the untrained arm now covers every input SNR in the slice,
at 10 seeds rather than 5.

### 5.1 The two metrics disagree in sign

Median paired difference across 16 records, Holm-corrected within each family; every cell in
this table is `different` (significant, and outside the margin) unless marked.

| comparison | input SNR | ΔSNR (dB) | `rpeak_integrity` | **QRS F1** |
|---|---|---|---|---|
| 10 ep − passive | −5 | +5.75 | −0.445 | **+0.341** |
| | 0 | +4.77 | −0.526 | **+0.532** |
| | 5 | +3.97 | −0.612 | **+0.574** |
| | 10 | +3.57 | −0.653 | **+0.667** |
| | 20 | +3.21 | −0.636 | **+0.657** |
| 50 ep − 10 ep | −5 | −2.07 | +0.283 | **−0.055** |
| | 0 | −1.58 | +0.293 | **−0.141** |
| | 5 | −0.53 (inconclusive) | +0.284 | −0.128 (inconclusive) |
| | 10 | −0.14 (equivalent) | +0.267 | −0.041 (inconclusive) |
| | 20 | +0.02 (equivalent) | +0.223 | −0.025 (inconclusive) |

**"STDP destroys R-peak preservation" (Phase 3 §5.6.2) is false.** Ten epochs of training
*raise* QRS F1 over the passive network by +0.34 to +0.67 at every input SNR. The same runs'
`rpeak_integrity` reports a drop of −0.45 to −0.65. Sensitivity rises +0.45 to +0.72, PPV rises
at every SNR from 0 dB up (+0.21 to +0.61; inconclusive at −5 dB), and RMS timing jitter falls
by 38–63 ms.

**§3's "reversal" is also reversed.** §3 found that 50 epochs recovers R-peak integrity relative
to 10 and concluded the canonical 10-epoch configuration "sits near the worst point for R-peak
integrity". On F1 it sits near the *best*: 50 epochs is worse than 10 at −5 and 0 dB, with PPV
lower at −5 to 5 dB (−0.07 to −0.17) and jitter higher by 6–18 ms at every SNR. Beyond 10
epochs, more training costs landmark fidelity *and* low-SNR ΔSNR.

### 5.2 Absolute values

Per-record means, then mean across the 16 records.

| SNR | epochs | ΔSNR (dB) | `rpeak_integrity` | QRS F1 | Se | PPV | jitter (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −5 | 0 | −0.58 | 1.000 | 0.220 | 0.193 | 0.355 | 89.4 |
| −5 | 10 | +5.12 | 0.533 | **0.481** | 0.553 | 0.450 | 59.6 |
| −5 | 50 | +3.04 | 0.836 | 0.433 | 0.506 | 0.386 | 67.8 |
| 0 | 0 | −3.50 | 1.000 | 0.177 | 0.161 | 0.293 | 86.2 |
| 0 | 10 | +1.34 | 0.484 | **0.572** | 0.631 | 0.550 | 47.0 |
| 0 | 50 | −0.16 | 0.823 | 0.457 | 0.546 | 0.403 | 58.9 |
| 5 | 0 | −6.57 | 1.000 | 0.149 | 0.143 | 0.240 | 87.3 |
| 5 | 10 | −2.53 | 0.430 | **0.637** | 0.658 | 0.661 | 29.3 |
| 5 | 50 | −2.98 | 0.765 | 0.549 | 0.643 | 0.491 | 50.8 |
| 10 | 0 | −9.10 | 1.000 | 0.126 | 0.128 | 0.179 | 82.4 |
| 10 | 10 | −5.53 | 0.401 | **0.647** | 0.660 | 0.667 | 23.7 |
| 10 | 50 | −5.69 | 0.682 | 0.613 | 0.667 | 0.589 | 42.5 |
| 20 | 0 | −11.26 | 0.999 | 0.133 | 0.130 | 0.204 | 83.4 |
| 20 | 10 | −7.92 | 0.394 | **0.657** | 0.664 | 0.693 | 27.8 |
| 20 | 50 | −7.98 | 0.647 | 0.628 | 0.680 | 0.609 | 36.3 |

At every input SNR, `rpeak_integrity` ranks the three networks in **exactly the reverse order**
of F1. The code says why. Its score is `matched / len(clean_peaks)` (`rpeak.py`) — recall only,
with no penalty for extra detections — and its prominence floor is a fraction of the output's own
standard deviation. A densely firing output (the passive network) is full of small local maxima,
one of which lands within 50 ms of almost every clean beat, so it scores perfectly while the
annotation-referenced detector finds nothing usable in it. A sparse output (the trained network)
has few maxima, misses some beats outright, and is penalised — even though its peaks are the
real ones.

### 5.3 Against doing nothing

> **Read with the correction at the top.** "Doing nothing" here is an unfiltered detector; with
> neurokit's standard cleaning it scores median F1 0.92–1.00 at every level, against a best
> network median of 0.71.

`scripts/r2_reanalysis/identity_baseline.py` scores the unprocessed input on the same windows
and noise draws, so each network can be paired against no processing. Full table in
`docs/r1_findings.md` §4.1. In brief: the passive network helps at −5 dB (+0.12) and destroys
recoverability at ≥ 5 dB (−0.55 to −0.76); the 10-epoch network helps at −5 and 0 dB (+0.41,
+0.36) and has no detectable effect above.

### 5.4 Consequences

- **There is no ΔSNR-versus-landmark trade-off in STDP.** Ten epochs dominates the passive
  network on both axes at every input SNR. The trade-off Phase 3 described was the saturated
  metric's rendering of a different effect: landmark recoverability depends on how sparsely the
  output fires (`docs/r1_findings.md` §5), and STDP from uniform initialisation moves the
  network into the sparse regime by depressing its weights.
- **Phase 4 and Phase 6 guidance must change.** "Target the joint (ΔSNR × rpeak) trade-off"
  optimises against a metric with the wrong sign. The joint target is (ΔSNR, QRS F1), and on
  current evidence it is not a trade-off at all over epochs.
- **The canonical 10-epoch configuration is well chosen**, for the opposite reason to the one
  §3 would have given.
- **Caveat.** Jitter comparisons against the passive network pair fewer than 16 records at
  ≥ 0 dB (11–14), because the passive output often yields no matched beat at all and so no
  jitter to average. That missingness is itself the finding, but those rows carry less weight.
