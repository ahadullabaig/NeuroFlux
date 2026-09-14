# R4b — The mechanism at R3's operating points

**Status:** complete. Roadmap next actions 13, 14 and 17.
**Code:** `scripts/r4b_mechanism/` (`training_fan_in.py`, `encoder_range.py`, `rate_optimum.py`,
`analyse_r4b.py`), on R3's pipeline (`scripts/r3_characterisation/r3_common.py`), which gained
two options for this stage: `encoder_range` and `fan_in`.
**Data:** `results/r4b_mechanism/*.csv.gz` (raw, gzipped), `results/r4b_mechanism/analysis/`.
**Reproduce:**

```bash
OMP_NUM_THREADS=1 uv run python scripts/r4b_mechanism/training_fan_in.py
OMP_NUM_THREADS=1 uv run python scripts/r4b_mechanism/encoder_range.py
OMP_NUM_THREADS=1 uv run python scripts/r4b_mechanism/rate_optimum.py --part width
OMP_NUM_THREADS=1 uv run python scripts/r4b_mechanism/rate_optimum.py --part fan-in
uv run python scripts/r4b_mechanism/analyse_r4b.py
```

About 15 minutes on 16 cores. 11,480 runs, 0 failures, every run `ok`, every output-rate
target reached (0.99–40.08 Hz achieved). Each prediction and its test is written in the grid
script's docstring and was written before that grid ran.

---

## Why this stage exists

R3 left three mechanism questions open, all created by its own results:

1. **Every mechanism number before R3 was measured with every layer at 40 Hz** (M1a, M3, M4),
   and R3 C3 found the ECG ceiling peaks at a sparser output. Does the mechanism hold there?
2. **The operating point a noisy input sets costs the ceiling up to 1.1 dB** on MIT-BIH at 0 dB
   (R3 §1.5), because the encoder's range is the noisy drive's own min and max. Is that a lever?
3. **Training lowers the ceiling at a held rate, and M3's column cosine does not explain why**:
   training *decorrelates* layer 1 while damaging it most (R3 §2.6). What does?

Two anchors confirm the pipeline agrees with what came before: `[32,16,8]` at 40 Hz gives the
1.92 dB MIT-BIH ceiling C1 and C2 measured, and `[512,512]` with 16 inputs per neuron gives
M4's +1.20 dB fan-in gain at 40 Hz exactly.

---

## 1. Training collapses effective fan-in (next action 17) — supported

`training_fan_in.py` re-runs R3 C2's training specs (rate-controlled `[32,16,8]`, noiseless
input, 10 seeds) and records each layer's weights; every run reproduces C2 exactly. A neuron's
**effective number of inputs** is the participation ratio of its weight column,
`PR = (Σ w)² / Σ w²` — N for equal weights, 1 for a single non-zero weight.

| MIT-BIH, rates held at 40 Hz | 0 epochs | 10 | 50 | 200 |
|---|---:|---:|---:|---:|
| ceiling (dB) | 1.92 | 1.34 | 1.13 | 1.06 |
| layer-1 effective inputs (of 32) | **24.3** | **5.1** | **3.0** | **2.6** |
| layer-2 effective inputs (of 16) | 12.1 | 9.6 | 4.8 | 4.5 |
| layer-1 column cosine | 0.76 | 0.45 | 0.37 | 0.38 |
| layer-1 intrinsic `S/N` (dB) | +3.69 | +0.96 | −0.52 | −0.84 |
| layer-2 intrinsic `S/N` (dB) | +0.98 | −1.56 | −1.99 | −1.88 |

Synthetic ECG: layer-1 effective inputs 24.4 → 5.6 → 2.7 → 2.2; layer-1 intrinsic `S/N`
+4.19 → −2.95 dB.

| prediction | MIT-BIH (16 records) | synthetic ECG (10 seeds) | verdict |
|---|---|---|---|
| P1: layer-1 effective inputs fall | 24.3 → 2.6, Wilcoxon p = 3 × 10⁻⁵ | 24.4 → 2.2, p = 0.002 | **supported** |
| P2: change in effective inputs tracks change in layer-1 intrinsic `S/N` (Spearman > 0) | ρ = **+0.79**, p = 4 × 10⁻¹¹, n = 48 | ρ = **+0.72**, p = 6 × 10⁻⁶, n = 30 | **supported** |
| P3: effective inputs predict it better than cosine | +0.79 against +0.11 (p = 0.45) | +0.72 against −0.70 | **met; decisively on MIT-BIH only** |

On synthetic ECG the cosine *falls* with the loss nearly as tightly as the effective fan-in does,
so there the two are confounded and only the MIT-BIH result separates them. For layer 2 the
effective-input correlation is +0.55 (p = 5 × 10⁻⁵) on MIT-BIH and not significant on synthetic
ECG (−0.31, p = 0.10).

**Reading.** STDP on a tiled input depresses most of each layer-1 neuron's synapses and keeps
two or three. A neuron averaging three Poisson trains has far noisier drive than one averaging
twenty-four, and recalibrating its gain to hold 40 Hz amplifies that noise along with the
signal. This is the other side of M4's fan-in trade-off: too many shared inputs correlate the
neurons (column cosine 3/4), too few leave each one noisy. Random dense weights sit on the first
side; trained weights fall off the second. The same quantity explains both the training loss
and M4's optimum.

---

## 2. The encoder range is a small lever on ECG and a large one on smooth signals (next action 14) — primary test not met

`encoder_range.py`: rate-controlled, untrained `[32,16,8]`; the encoder's `[v_min, v_max]` from
the drive's min/max (R3's choice, the reference), its 0.1st/99.9th percentiles, the extremes of a
zero-phase 40 Hz low-pass copy, or the clean signal's extremes (not deployable; the bound). The
encoder always sees the unfiltered noisy drive.

Output SNR change against min/max, median over units [95% BCa CI]; all significant after Holm
unless marked:

| input SNR (dB) | −10 | −5 | 0 | 5 | 10 | 20 | noiseless |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIT-BIH, low-pass | +0.13 | +0.28 | **+0.35** [+0.27, +0.47] | +0.32 | +0.21 | +0.06 | +0.03 (n.s.) |
| MIT-BIH, percentile | +0.02 | +0.07 | +0.13 | +0.17 | +0.15 | +0.10 | +0.08 |
| MIT-BIH, oracle bound | +0.30 | +0.58 | +0.69 | +0.55 | +0.29 | +0.06 | 0 |
| synthetic ECG, low-pass | +0.14 | +0.22 | +0.26 | +0.18 | +0.12 | +0.09 | +0.03 (n.s.) |
| sine, low-pass | +0.31 | +0.56 | **+1.02** | **+1.29** | **+1.37** | +0.48 | 0.00 |
| sine, oracle bound | +0.58 | +1.54 | +2.67 | +3.13 | +2.48 | +0.85 | 0 |

**Primary test (pre-registered): not established.** On MIT-BIH at 0 dB the low-pass range raises
output by +0.35 dB — significant, and without cost to the noiseless ceiling — but below the
+0.5 dB the test required, and TOST places it inside the ±0.5 dB margin. The percentile range
gains only +0.13.

What the grid does show:

- **The range explains about 60% of the operating-point cost, not all of it.** Even the oracle
  range recovers only 0.69 of the 1.11 dB that R3 §1.5 attributed to the noisy operating point at
  0 dB on MIT-BIH. The encoder-rate and layer-gain calibrations, which are still fitted to the
  noisy drive in every arm here, are the obvious candidates for the rest; that is untested.
- **A deployable range recovers a third to three quarters of what the oracle does** on ECG
  (low-pass: 43–72% of the oracle's gain between −10 and 10 dB on MIT-BIH, 36–41% on synthetic
  ECG).
- **On a smooth signal it is a real lever.** On the sine, the low-pass range adds 1.0–1.4 dB
  between 0 and 10 dB input, significant and not equivalent to zero, with no cost to the
  ceiling. The sine's whole band sits below the filter's cutoff, so the low-passed range is
  nearly the clean one; ECG's QRS complexes carry energy the filter's edge begins to trim.
- **Clipping to a narrower range does not hurt at high input SNR** in any arm: every method is
  equivalent to min/max at the noiseless input.

---

## 3. The mechanism at the rate optimum (next action 13) — both predictions supported

### 3.1 Input width matters more at a sparse output

`rate_optimum.py --part width`: `[N_in, 16, 8]`, rate-controlled, noiseless, output held at 5 Hz
or 40 Hz (hidden layer and encoder at 40 Hz). MIT-BIH median ceiling (dB):

| N_in | 8 | 16 | 32 | 64 | 128 | 256 | slope per doubling [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| output 5 Hz | 1.44 | 1.88 | 2.40 | 2.56 | 3.11 | **3.17** | +0.279 [+0.245, +0.343] |
| output 40 Hz | 1.31 | 1.69 | **1.92** | 2.09 | 2.23 | 2.33 | +0.198 [+0.164, +0.270] |

Per-record slope difference, 5 Hz minus 40 Hz: **+0.078 [+0.033, +0.128], Wilcoxon p = 0.008,
n = 16 — supported.** A sparse output extracts more from a wider tiled input, and at 256 inputs
reaches 3.17 dB. The 40 Hz slope here (+0.20 dB/doubling, noiseless input, encoder
rate-controlled) is steeper than M1a's tiled +0.10 (0 dB input, encoder uncontrolled), so M1a's
figure is specific to its conditions; both are far below the +3 dB/doubling of independent
channels.

### 3.2 The fan-in intervention survives — and grows — at the rate optimum

`rate_optimum.py --part fan-in`: `[N, N]`, one LIF layer as in M4, rate-controlled, noiseless,
layer held at 1–40 Hz. Dense weights against exactly K inputs per neuron (M4's optima).

MIT-BIH median ceiling (dB), dense / sparse fan-in:

| layer rate (Hz) | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| `[128,128]`, K = 8 | 0.88 / 2.01 | 2.31 / 3.20 | 3.19 / 4.24 | 3.44 / 5.21 | 3.68 / **5.34** | **3.89** / 4.59 |
| fan-in gain, paired median | +1.09 | +0.65 | +1.27 | +1.33 | +1.22 | +0.81 |
| `[512,512]`, K = 16 | 0.76 / 1.78 | 2.11 / 3.32 | 3.15 / 4.51 | 3.57 / 5.67 | **4.06** / **5.77** | 3.93 / 5.40 |
| fan-in gain, paired median | +1.01 | +0.94 | +1.67 | +1.78 | +1.85 | +1.20 |

Every gain is significant (Wilcoxon p ≤ 6 × 10⁻⁵).

**Pre-registered test** — sparse fan-in at its best rate against dense at its best rate, paired
by record, Holm across the two widths:

| network | sparse best | dense best | gain [95% CI] | p_holm | verdict |
|---|---|---|---|---|---|
| `[128,128]`, K = 8 | 20 Hz | 40 Hz | **+1.10** [+0.73, +1.57] | 6 × 10⁻⁵ | **supported** |
| `[512,512]`, K = 16 | 20 Hz | 20 Hz | **+1.85** [+1.38, +2.26] | 6 × 10⁻⁵ | **supported** |

- **The intervention does not depend on the 40 Hz operating point; it is larger away from it.**
  At 40 Hz the `[512,512]` gain is +1.20 dB, M4's exact figure; at 10–20 Hz it is +1.8 dB.
- **5.77 dB is the highest ceiling measured on MIT-BIH in this project** — `[512,512]`, 16
  inputs per neuron, 20 Hz — against 1.92 dB for the shipped `[32,16,8]` at 40 Hz: five
  times the signal-to-noise power ratio (`S/N` 2.78 against 0.56) from architecture and
  operating point alone.
- **The rate optimum depends on the architecture as well as the signal.** One-layer `[N, N]`
  networks peak at 10–20 Hz; the `[32,16,8]` output peaks near 5 Hz (R3 §3.2); the sine and EEG
  peak at 20–40 Hz (R3 §3.3).

---

## 4. What R4b does to the paper

**The mechanism sentences** held back from the framing statement can be restated in the form
the data support:

> Each stage's intrinsic SNR is limited by the error its neurons share, which is set by how many
> inputs they draw on and how much those inputs overlap: a dense non-negative random draw makes
> every pair of weight columns 3/4 similar, and too few effective inputs leaves each neuron
> noisy. Restricting each neuron to a fixed fan-in of 8–16 lifts the ceiling at every output
> rate tested, by up to 1.85 dB (5.77 dB on real ECG); unsupervised STDP instead collapses
> layer-1 effective fan-in from 24 inputs to 3, which is why learning lowers the ceiling.

**What changes from M3 and M4.** M3's column-cosine account is the dense-weight half of a
fan-in trade-off, and it does not predict a trained layer's noise; the effective fan-in does.
M4's intervention generalises across output rate and is larger at the optimum than M4 measured.

**What does not change.** The ceiling is still flat in input quality at every configuration
here, and the best architecture found is still 5.77 dB on MIT-BIH — a characterisation of a
limit, not a competitive denoiser.

**Encoder range** is reported as a partial lever: small on ECG (+0.35 dB at 0 dB input, below
its pre-registered bar), large on smooth signals (+1.0–1.4 dB on the sine), with no cost to the
ceiling.

**Still open.** Whether a fan-in constraint *during* training prevents the collapse (and so
lets STDP help rather than hurt); the calibration share of the operating-point cost that the
encoder range does not reach; and the confound between fan-in and cosine on synthetic ECG.
