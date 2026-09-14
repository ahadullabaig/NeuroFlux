# R5 — Generality: encoders, readouts, and the QRS detector

**Status:** complete. Roadmap stage R5; next actions 12 and 16.
**Code:** `scripts/r5_generality/` (`encoders_decoders.py`, `qrs_detectors.py`, `analyse_r5.py`,
`figures_r5.py`), on R3's pipeline (`scripts/r3_characterisation/r3_common.py`), which gained
temporal and delta encoders, spike-rate and latency readouts, a delay-compensated SNR, and an
output hook. Core gained `calibrate_delta_threshold` / `delta_encoder_rate_hz`, and
`DeltaEncoder`'s docstring was corrected (§0).
**Data:** `results/r5_generality/*.csv.gz` (raw, gzipped), `results/r5_generality/analysis/`,
Table II in `results/r5_generality/figures/table2_generality.tex`.
**Reproduce:**

```bash
uv run python scripts/r5_generality/encoders_decoders.py   # ~10 min on 16 cores
uv run python scripts/r5_generality/qrs_detectors.py       # ~7 min
uv run python scripts/r5_generality/analyse_r5.py
uv run python scripts/r5_generality/figures_r5.py
```

13,900 runs, 0 failures. Every R5a run is `ok` under every readout; 10 of R5b's 5,920 runs are
`silent`, all shipped 10-epoch networks that were already silent in R3 C1. Each test is written
in its grid script's docstring, fixed before that grid ran, together with what a small pilot had
shown. Two built-in
checks passed: the 1,330 Poisson runs R5a shares with R3 C1 reproduce it exactly, and all 5,920
R5b runs reproduce their stored R3 rows exactly. The analysis reproduces byte for byte from the
gzipped data and under a different `PYTHONHASHSEED`.

---

## Why this stage exists

The roadmap's R5 asks whether the ceiling is a property of an architecture class or of one
encoder and one decoder. Finding 5 had answered it on trained, uncalibrated networks, with the
delta rows degenerate and a time-to-first-spike code read by rate decoders that cannot see it.
Separately, every QRS F1 in R1–R3 came from one detector run on the unfiltered signal, and R3
§1.6 had already shown that detector's standard filter beats every network on the input.

---

## 0. What changed before any R5 number was taken

**Three encoders at one operating point.** All encoders fire at 40 Hz. Poisson is calibrated as
in R3. The temporal encoder's rate is its window, so a 24 ms window (six 4 ms steps, six latency
levels) gives 41.7 Hz. The delta encoder's threshold is bisected to 40 Hz on each drive by the new
`calibrate_delta_threshold`; next action 12 asked for exactly this. Achieved encoder rates were
38.9–40.7 Hz (Poisson), 41.6–41.7 Hz (temporal) and 38.7–40.6 Hz (delta).

**A readout that can see a latency code.** `latency_readout` reads any stage as the population's
mean first-spike step in each encoder window. A later stage's window is shifted by its delay, which
is estimated from spikes alone, and the value is held causally from the window's close. On every
run the delay came out as 0 steps: at 40 Hz a calibrated LIF layer answers a volley within the
step it arrives.

**A delay-compensated SNR.** A pilot showed that readouts trail the signal by 4–40 ms, and at
high SNR that delay costs as much as the noise: a latency readout 40 ms late caps a 2 Hz sine
near 6.5 dB on its own. Comparing encoders on causal SNR would compare delays. Every row therefore
also carries `snr_best_lag_db`, the affine SNR after advancing the output by its best lag in
0–64 ms. Every lag is scored on the same samples, and only positive-scale fits count, so an
inverted or cycle-late match cannot win. It is the primary metric for R5a's comparisons. Causal
SNR, the metric of every R3 table, is reported beside it. The median best lag of every cell is
0–52 ms and matches its readout: about W/2 for a boxcar, below τ for an exponential, about 40 ms
for latency. Individual runs reach the 64 ms bound in at most 0.2% of Poisson runs and 0.7% of
latency runs, but in 8–17% of delta runs, whose SNR barely changes with lag. A live output with
no positive-scale fit at any lag scores 0 dB, the value a scale-constrained fit takes; this
happens only for delta on the sine.

**The `DeltaEncoder` docstring was wrong.** It advised pairing a "positive" and a "negative"
encoder for a signed code. A polarity channel's baseline moves only when that channel fires, so it
reports new extremes: on a sine it fires 8 times in the first cycle and never again, and the pair
cannot be integrated back into the input. The docstring now says so, and a test pins it. A signed
delta code was therefore not tested (§3).

---

## 1. R5a — encoders and decoders

**Grid** (`results/r5_generality/encoders_decoders.csv.gz`, 7,980 runs): untrained `[32,16,8]`,
hidden layer at 40 Hz, output at 40 Hz or 5 Hz (R3 C3's MIT-BIH optimum). Poisson, temporal and
delta encoders; MIT-BIH (16 records), synthetic ECG, EEG and the 2 Hz sine, 10 seeds; Gaussian
noise at −10 to +40 dB and noiseless. Every stage is read with population decoders (τ 10, 20,
40 ms), spike-rate decoders (window 20, 40, 80 ms) and, for the temporal encoder, the latency
readout.

**Table II** (noiseless ceiling, dB: delay-compensated, causal in parentheses, each readout at its
best window; median over records or seeds):

| output | encoder | readout | MIT-BIH | synthetic ECG | EEG | 2 Hz sine |
|---|---|---|---:|---:|---:|---:|
| 40 Hz | Poisson | population | 2.55 (1.98) | 2.74 (1.52) | 3.58 (2.29) | 11.45 (6.66) |
| | | **spike rate** | **2.86 (1.90)** | **3.33 (1.92)** | **4.23 (1.14)** | **12.63 (6.50)** |
| | temporal | **latency** | 2.28 (0.72) | 2.81 (0.00) | 3.55 (0.71) | 12.95 (5.66) |
| | | population / spike rate | 0.03 / 0.05 | 0.01 / 0.03 | 1.16 / 1.78 | 0.03 / 0.09 |
| | delta | population | 0.53 (0.39) | 1.42 (1.01) | 2.92 (0.08) | 0.00 (0.05) |
| | | spike rate | 0.61 (0.18) | 1.61 (0.97) | 4.50 (0.07) | 0.00 (0.04) |
| 5 Hz | Poisson | population | 2.71 (2.40) | 3.94 (3.23) | 2.53 (0.56) | 5.68 (4.21) |
| | | **spike rate** | **3.08 (1.77)** | **5.30 (2.87)** | **3.23 (0.02)** | **6.21 (4.33)** |
| | temporal | **latency** | 0.34 (0.09) | 0.79 (0.03) | 2.19 (0.35) | 0.89 (0.88) |
| | delta | spike rate | 0.84 (0.59) | 1.68 (0.62) | 6.06 (2.02) | 0.01 (0.01) |

Bold: Poisson's best readout, and the latency readout built for the temporal code (its best
readout everywhere except EEG at 5 Hz, where a spike-rate readout scores 2.93). The LaTeX version,
with every row and the test marks, is `table2_generality.tex`.

### 1.1 T1 — the ceiling is not specific to rate coding

Pre-registered: temporal coding *escapes* the ceiling on a signal if its best readout's noiseless
ceiling exceeds Poisson's by a significant median of at least +3 dB; the ceiling is specific to rate
coding only if that happens on three of four signals. At a 40 Hz output (temporal's best readout
is latency on every signal):

| temporal − Poisson | median [95% BCa] | p (Holm, 4) | TOST ±0.5 dB |
|---|---|---:|---|
| MIT-BIH | −0.76 [−1.13, +0.24] | 0.55 | not equivalent |
| synthetic ECG | −0.54 [−0.73, −0.07] | 0.078 | not equivalent |
| EEG | −0.89 [−1.24, −0.05] | 0.11 | not equivalent |
| 2 Hz sine | +0.21 [−0.47, +0.60] | 0.55 | **equivalent** |

**Temporal coding escapes on 0 of 4 signals: the ceiling is not specific to rate coding.** A
time-to-first-spike code with a readout built for it is equivalent to Poisson on the sine. On the
other three signals its median is 0.5–0.9 dB lower, neither significant after Holm nor equivalent,
so it may be slightly worse there and is not better. On causal SNR the gap is wider (MIT-BIH 0.72
against 1.90 dB), because the latency readout waits for its window to close.

It is also more fragile. **A latency code needs every neuron to fire every window**: with the
output held at 5 Hz, temporal falls to 0.34 dB on MIT-BIH, 0.79 on synthetic ECG and 0.89 on the
sine. That is 2.7–5.3 dB below Poisson, significant on those three signals, and on EEG its best
readout becomes a rate readout.

Delta coding, reported but not part of the decision, is significantly lower than Poisson on both
ECGs and on the sine at both rates. It is higher only on EEG at 5 Hz (+2.84 dB, p_holm 0.027).
That exception is not a better code (§1.4).

### 1.2 T2 — flat for every encoder

Slope of output SNR (best readout, compensated) on input SNR over 10–40 dB, median over units:
every one of the 24 (signal, rate, encoder) cells has an interval below 1, C1's criterion. The
largest slopes are +0.16 (delta, EEG, 5 Hz), +0.14 (temporal, sine, 40 Hz) and +0.13 (Poisson,
sine, 40 Hz); MIT-BIH is +0.003 to +0.018 for every encoder. **The flatness that defines the
ceiling holds for all three codes and all four signals.**

The criterion is weak for delta on EEG. Its output is flat over 30–40 dB but rises 3 dB from 10 to
30 dB (1.02 → 3.92 at 40 Hz), and the fitted slope stays well below 1 because the change is spread
over 30 dB.

### 1.3 T3 — decoder type: the pre-registered criterion is not met, and why

Pre-registered: the ceiling is not a population-decoder artefact if no spike-rate readout is
significantly higher than its matched population readout by more than 0.5 dB, where a W ms boxcar
is matched to a W/2 ms exponential (the same noise bandwidth). Holm across all 32 comparisons.

Two comparisons fail it, both on MIT-BIH at a 40 Hz output: a 20 ms boxcar beats a 10 ms
exponential by +0.65 dB [+0.53, +0.75], and a 40 ms boxcar beats a 20 ms exponential by +0.69
[+0.59, +0.82] (p_holm 0.001 each). **The criterion is not met.** What it measured is narrower than
"decoder type":

- **At their best windows the two decoders are equivalent on real ECG**: best spike rate minus
  best population +0.10 dB [−0.03, +0.33] at 40 Hz and +0.31 [+0.10, +0.45] at 5 Hz, both
  equivalent within ±0.5 dB.
- **The boxcar's advantage is its phase, and it exists only after delay compensation.** A boxcar
  delays every frequency equally, so one lag undoes it; an exponential's delay varies with
  frequency. On causal SNR the matched pair is level on MIT-BIH (1.90 against 1.92 dB).
- On the synthetic signals the best boxcar is 0.53–1.30 dB above the best exponential with
  intervals excluding zero, but no comparison can be significant. With 10 seeds the smallest
  exact Wilcoxon p is 0.002, so a Holm family of 32 floors every synthetic p at 0.057. The Holm
  family was not specified in the pre-registration; the analysis fixed it at all 32 before the
  data existed.

**No decoder removes the ceiling (T2), and on MIT-BIH the decoder type does not change its
height.** Where a boxcar helps, it helps by up to 1.3 dB, through phase rather than noise.

### 1.4 T4 — delta coding at a calibrated threshold

Every delta run is `ok` under every readout (threshold 0.047–6.62), so **next action 12 is
closed**: the degenerate delta rows of Finding 5 were an uncalibrated threshold. They still carry
almost nothing a rate readout can use. An absolute-mode delta encoder fires in proportion to
`|dx/dt|`, not `x`, and reconstructs:

- **nothing on the sine**: 0.00 dB compensated, anti-correlated at every lag;
- **0.5–1.7 dB on the ECGs**, where the slope's magnitude peaks at the QRS complex, as the signal
  does;
- **up to 6.06 dB on EEG, but only near noiseless input.** The synthetic EEG is three pure
  sinusoids, and a deterministic threshold crossing phase-locks to them. At 10 dB input delta
  gives 1.02 dB at 40 Hz and 0.49 at 5 Hz, against Poisson's 2.61 and 2.87.

**The layers behind a delta encoder could not be held at 40 Hz on ECG** (median 30.7 Hz on
MIT-BIH, 30.0 on synthetic ECG; 37–39 on sine and EEG); the layer-1 gain sat at the
calibration's upper bound in 98% of ECG runs. A probe on record 100 shows why: 44% of its delta
volleys arrive on consecutive steps, a refractory LIF neuron cannot follow them, and layer 1
saturates at 33 Hz whatever the gain. The delta rows are therefore at a lower rate than Poisson's, which is one more reason they
stay out of the T1 decision.

### 1.5 Post hoc: readout delay is part of every causal ceiling so far

Not pre-registered. Every ceiling in M3, R3 and R4b is causal. For Poisson, the best
delay-compensated readout against the best causal one:

| output | MIT-BIH | synthetic ECG | EEG | 2 Hz sine |
|---|---:|---:|---:|---:|
| 40 Hz, best causal | 1.98 (population 40 ms) | 1.92 | 2.72 | 8.40 |
| 40 Hz, best compensated | 2.86 (spike rate 40 ms) | 3.33 | 4.23 | 12.63 |
| gain | **+0.88** | +1.41 | +1.51 | **+4.23** |
| 5 Hz gain | +0.68 | +1.89 | +2.11 | +1.88 |

Removing a fixed delay is free offline, so the offline ceiling is 0.7–4.2 dB higher than the causal
one: **R3's 1.92 dB MIT-BIH ceiling is a causal number, and about 0.9 dB of it is delay.** The loss
is largest on the signals whose energy sits where a 10–40 ms delay is a large phase error: EEG's
10–40 Hz rhythms, and the sine at high SNR. Two R3 statements need that qualifier:

- **§2.3, "no decoder window lifts the ceiling"**, holds for causal readouts. Compensated, a 40 ms
  boxcar sits 0.88 dB above the best causal window on MIT-BIH.
- **§1.2, "height signal-dependent (sine 7.48 dB)"**: the sine's compensated ceiling is 12.6 dB.
  The ordering of signals survives (sine ≫ EEG > ECG), but the gap is partly delay.

The ceiling is still flat in input SNR under the compensated metric (T2), and nothing here touches
the cascade law, which R3 verified on causal intrinsics measured and applied consistently.

---

## 2. R5b — the QRS conclusions and the detector

**Grid** (`results/r5_generality/qrs_detectors.csv.gz`, 5,920 runs): R3 C3's output-rate axis on
MIT-BIH (controlled `[32,16,8]`, output 0.25–80 Hz, noiseless, 0 and 10 dB) and R3 C1's shipped
arms (passive and 10-epoch networks at −5 to +20 dB), each run reproducing its stored R3 row. The
input and the output are scored with `neurokit`, `pantompkins1985` and `hamilton2002`, raw and after
`ecg_clean` with the detector's own cleaning method: six variants. A 4-run pilot, written into the
script before the grid ran, had shown a dense 40 Hz output scoring F1 0.00 raw and 0.90 after
cleaning.

**What each detector gets from the noisy input** (median F1 over records):

| input | −5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---:|---:|---:|---:|---:|
| `neurokit`, raw / cleaned | 0.06 / **0.90** | 0.22 / **0.98** | 0.89 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| `pantompkins1985`, raw / cleaned | 0.49 / 0.57 | 0.51 / 0.71 | 0.51 / 0.89 | 0.55 / 0.97 | 0.87 / 0.99 |
| `hamilton2002`, raw / cleaned | 0.47 / 0.65 | 0.47 / 0.80 | 0.48 / 0.93 | 0.60 / 0.92 | 0.85 / 0.96 |

### 2.1 Q1 — "beat detection needs a sparse output" depends on the missing filter

R3 C3, pre-registered then and confirmed on the raw `neurokit` detector: F1 at 10 dB is higher with
the output at 1 Hz than at 40 Hz. Re-tested under all six variants (Holm across six):

| variant | 40 Hz | 1 Hz | 1 Hz − 40 Hz [95% BCa] | p (Holm) | survives |
|---|---:|---:|---|---:|---|
| `neurokit`, raw | 0.04 | 0.77 | +0.72 [+0.61, +0.82] | 0.0002 | yes |
| `neurokit`, cleaned | **0.81** | **0.81** | **+0.01 [−0.14, +0.10]** | 0.94 | **no** |
| `pantompkins1985`, raw | 0.49 | 0.71 | +0.24 [+0.10, +0.37] | 0.038 | yes |
| `pantompkins1985`, cleaned | 0.54 | 0.77 | +0.11 [+0.04, +0.33] | 0.047 | yes |
| `hamilton2002`, raw | 0.48 | 0.65 | +0.13 [−0.02, +0.32] | 0.077 | no |
| `hamilton2002`, cleaned | 0.56 | 0.79 | +0.15 [+0.07, +0.26] | 0.038 | yes |

**It survives 4 of 6 variants, so it is not detector-independent.** The 0.77-against-0.04 headline
is almost entirely the unfiltered detector. With neurokit's standard pipeline a dense 40 Hz output
is as detectable as a 1 Hz one (0.81 each); the curve over rates dips between them instead
(0.47 at 20 Hz). A sparse output keeps a small advantage of +0.11 to +0.15 F1 under the two
detectors that are weaker on the input.

### 2.2 Q2 — "training raises F1 over the passive network" does not survive

R2 §5 and R3 C1: 10 epochs of STDP raise F1 over the passive shipped network. Pre-registered to
survive a variant only if positive and significant at all five inputs (Holm across 30):

| 10 epochs − passive | −5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---:|---:|---:|---:|---:|
| `neurokit`, raw | +0.30 | +0.40 | +0.59 | **+0.64** | **+0.64** |
| `neurokit`, cleaned | +0.21 | +0.18 | +0.16 | +0.06 | −0.01 |
| `pantompkins1985`, raw / cleaned | +0.00 / −0.01 | +0.10 / +0.05 | +0.14 / +0.10 | +0.16 / +0.09 | +0.26 / +0.16 |
| `hamilton2002`, raw / cleaned | −0.05 / +0.02 | +0.04 / +0.10 | +0.10 / +0.11 | +0.08 / +0.14 | +0.11 / +0.14 |

Bold: p_holm < 0.05. **It survives 0 of 6 variants.** These are R3 C1's shipped networks at true
Gaussian input SNR, not R2's own flicker windows, so this re-tests R2's conclusion rather than
re-scoring its runs. On the original detector the direction replicates and the effect is large,
significant after correction at 10 and 20 dB. With cleaning, the passive network's F1 rises from
0.05–0.16 to **0.33–0.78** while the trained network's barely moves (0.47–0.69); at 20 dB the
passive network is level with the trained one (0.78 against 0.69).

**R2's "the passive network destroys QRS recoverability at ≥ 5 dB" shrinks to "harms it".** The
near-zero F1 was the unfiltered detector failing on a dense output. Cleaned, the passive output
still scores 0.57–0.78 at 5–20 dB against the cleaned input's 1.00, and the trained network's
0.66–0.69 is below it too (§2.3).

### 2.3 Q3 — no network condition beats a standard filter, under any detector

At each input and detector, the cleaned noisy input against the best of every network condition
(any output rate, either shipped arm, raw or cleaned output). **The network is higher in none of
the 18 cells; the cleaned input is significantly higher in 16.** The two exceptions are
`pantompkins1985` at −5 dB (0.57 against 0.54) and 0 dB (0.71 against 0.69). At 10 dB the cleaned
input scores 1.00 / 0.97 / 0.92 across the three detectors, against the best network's 0.81 /
0.78 / 0.79. R3 §1.6's conclusion now holds for three detectors, not one.

### 2.4 Q4 — what cleaning the output does

Cleaning rescues exactly the outputs the raw detector failed on and leaves the rest alone.
`neurokit` at 10 dB: 40 Hz +0.78, 20 Hz +0.32, 10 Hz +0.21, the passive shipped network +0.60;
1 Hz +0.02, 0.5 Hz +0.00, the trained network +0.01. The beats in a dense output are recoverable
once `ecg_clean`'s 0.5 Hz high-pass and powerline filter have run. Which component matters was
not isolated. Both earlier conclusions were comparisons in which one side had its beats hidden
from an unfiltered detector: the dense output, or the passive network, which fires densely
(R1 §5).

---

## 3. Scope: what the claim covers

R5's acceptance criteria asked for the ceiling across at least two encoders, two decoders and
three signal types, and an explicit statement of scope.

**Demonstrated for:** feed-forward LIF networks with non-negative weights, untrained, their
layers held at a calibrated rate, fed a scalar signal tiled across the input population and read
by pooling the output population. Within that class the ceiling:

- is **flat in input quality** for Poisson rate, time-to-first-spike and delta codes, on real ECG,
  synthetic ECG, EEG and a sine, at output rates of 40 and 5 Hz (T2, 24 of 24 cells);
- is **not lifted by temporal coding**: 0 of 4 signals (T1);
- is **not a population-decoder artefact on real ECG**: best boxcar and best exponential are
  equivalent. A boxcar raises the offline ceiling by up to 1.3 dB on synthetic signals, through
  phase (T3, criterion not met as written, §1.3).

**Not covered:**

- **A signed delta code.** It needs a baseline shared by both polarities, which `DeltaEncoder`
  does not provide (§0), and sign-separated or inhibitory pathways, which a non-negative network
  does not have. The absolute-mode code that was tested carries `|dx/dt|`.
- **Temporal codes at other window lengths.** 24 ms gives six latency levels at dt = 4 ms, and
  finer levels need a longer window, which samples more slowly. Only the rate-matched window was
  run.
- **Non-tiled input, depth and trained networks** are not re-run here. M1a covered independent
  and delay-line inputs at 40 Hz on MIT-BIH, 10 seeds; R3 C2 covered depth and training.
- **Inhibition, recurrence, learned readouts, and non-LIF neurons.**

---

## 4. What R5 does to the paper

1. **Table II exists**, and it answers the roadmap's headline question in the negative: the
   ceiling does not disappear under temporal coding. Finding 5's encoder × decoder table is
   superseded. Its "temporal is worse" survives only in part: on causal SNR, and at a sparse
   output.
2. **Every ceiling number the manuscript quotes must say "causal".** The offline, delay-compensated
   ceiling is 0.7–4.2 dB higher (§1.5). The framing statement's "not lifted … by any decoder window"
   holds for causal readouts only. Proposed wording, for the author's review:

   > … it is not lifted by a wider output population at a dense output, by any causal decoder
   > window, by a time-to-first-spike code, or by unsupervised learning … Its height is set by the
   > signal, the depth, the output firing rate and the readout's delay.

3. **Every QRS statement in the manuscript is detector-scoped or dropped.** Two of R3's and R2's
   beat-recovery conclusions reverse or shrink under a standard filter (§2.1–§2.2); only "a
   standard filter beats every network" survives every detector (§2.3). The QRS rows of Table I
   and the F1 panel of Fig. 3 were scored with the unfiltered detector and must be labelled so,
   or re-scored with cleaning. The paper's case does not depend on them: the ceiling is a
   reconstruction result.
4. **Next actions closed: 12 (delta threshold) and 16 (QRS with and without `ecg_clean`).**
   Opened: whether a signed step-forward delta code through sign-separated pathways escapes the
   ceiling — a new architecture, and a limitation for the manuscript rather than an R5 item.
