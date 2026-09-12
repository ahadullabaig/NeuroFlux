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

## 3. Re-analysis: the Phase 3 claims hold, and get stronger

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
