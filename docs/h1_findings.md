# H1 — What the ceiling costs in hardware (MCSoC 2027)

**Status:** pilot B and grid B2 complete. **The frame decision is met:** cost against fidelity
is the MCSoC paper's frame (§3). Pilot A (weight precision and device faults) is designed and
pre-registered but has not run (§5).
**Why this stage exists:** the ICEE 2026 paper already claims the flat ceiling, the cascade law,
the K = 8–16 fan-in rule and STDP's collapse of fan-in. The next paper goes to the NeuroCore
summit at IEEE MCSoC 2027, an architecture venue whose reviewers will ask what a design point
costs in hardware. None of the stages so far reports any cost.
**Source data:**
- Pilot B ran no new simulations. It used committed rows from R3 C2 and C3 and from R4b, with
  a noiseless input and the 20 ms decoder: 76 design points on MIT-BIH and 60 on the synthetic
  signals.
- Grid B2 (`results/h1_hardware_cost/b2_grid.csv.gz`) is 1,920 runs with 0 failures, every run
  `ok`.

**Outputs:**
- `results/h1_hardware_cost/analysis/pilot_b_{points,fronts,contrasts}.csv`
- `results/h1_hardware_cost/analysis/b2_{cells,tests,front,post_hoc}.csv`
- `results/h1_hardware_cost/figures/pilot_b_front.pdf` and `b2_front.pdf`
- the raw grid, committed gzipped as `results/h1_hardware_cost/b2_grid.csv.gz`

**Reproduce:**
`uv run python scripts/h1_hardware_cost/pilot_b_cost.py`, then
`OMP_NUM_THREADS=1 uv run python scripts/h1_hardware_cost/b2_grid.py` (about 6 min on 16 cores),
then `uv run python scripts/h1_hardware_cost/analyse_b2.py`.

---

## 1. Pilot B — the cost of every measured design point

### 1.1 Cost model

Everything is per second of signal, computed from each run's measured encoder rate and layer rates.
The script docstring gives the formulas.

- **Synapses.** A fixed fan-in K applies to the first weight matrix only, as in R4b.
- **Synaptic operations.** One weight read per presynaptic spike per synapse. This is the work
  unit of an event-driven chip.
- **Spikes.** Address-event traffic, including the encoder's.
- **LIF neurons.**

A clock-driven chip touches every synapse and every neuron at every 4 ms step. For it, cost is
the synapse and neuron counts, and firing rates do not matter.

### 1.2 The front on MIT-BIH

A design point is on the front if no other point has a higher ceiling at no more than 5% more
cost, and no point that is at least 5% cheaper has a ceiling as high. The tolerance is there
because calibration leaves measured rates about 1% off target.

| front point | ceiling (dB) | synaptic ops/s | synapses | spikes/s |
|---|---:|---:|---:|---:|
| `[32,8]`, 40 Hz | 2.75 | 1.02 × 10⁴ | 256 | 1,597 |
| `[128,128]`, K = 8, 20 Hz | 5.34 | 4.10 × 10⁴ | 1,024 | 7,692 |
| `[512,512]`, K = 16, 20 Hz | 5.77 | 3.28 × 10⁵ | 8,192 | 30,710 |
| *reference: `[32,16,8]`, 40 Hz* | *1.92* | *2.56 × 10⁴* | *640* | *2,243* |

The same three points form the synapse front. The spike front has ten points: sparse-output
`[N_in,16,8]` networks, `[32,16,8]` with every layer at 5 Hz, `[32,8]`, and the fan-in
networks at 2–20 Hz.

- **Every point on the synaptic-operation and synapse fronts is a one-LIF-layer network.** The
  `[32,16,8]` family is dominated at every rate, width and training level. The one-layer
  `[32,8]` beats the rate-controlled `[32,16,8]` by 0.91 dB [0.79, 1.09] with 40% of its
  synaptic work. A second layer costs twice: 2.5× the synaptic operations for −0.91 dB, and a
  third costs 1.4× for another −0.36 dB.
- **The front has a knee at `[128,128]`, K = 8.** Up to the knee, the ceiling rises by 2.6 dB
  over a 4× increase in synaptic work. Past it, `[512,512]` with K = 16 buys +0.64 dB
  [+0.22, +0.98] for 8× the synaptic operations, 8× the synapses and 4× the spikes.

### 1.3 Exchange rates

Other minus reference, paired by record (n = 16):

| lever | Δ ceiling (dB) [95% CI] | synaptic ops | spikes | dB per doubling of synaptic ops |
|---|---|---:|---:|---:|
| output 5 Hz vs 40 Hz, `[32,16,8]` | +0.65 [+0.02, +1.27] | ×1.00 | ×0.87 | — |
| every layer 10 Hz vs 40 Hz | +0.28 [−0.13, +0.68], equivalent | ×0.85 | ×0.68 | — |
| output width 64 vs 8 | +0.08 [+0.06, +0.09], equivalent | ×2.41 | ×2.00 | +0.06 |
| input width 256 vs 32, output 40 Hz | +0.45 [+0.33, +0.56] | ×6.61 | ×5.00 | +0.17 |
| input width 256 vs 32, output 5 Hz | +0.62 [+0.46, +0.74] | ×6.61 | ×5.58 | +0.23 |
| depth, 2 vs 1 LIF layer | −0.91 [−1.09, −0.79] | ×2.50 | ×1.40 | −0.69 |
| depth, 3 vs 2 LIF layers | −0.36 [−0.40, −0.30], equivalent | ×1.40 | ×1.29 | −0.73 |
| 10 STDP epochs vs none, held rate | −0.49 [−0.60, −0.39] | ×1.00 | ×1.00 | — |
| `[128,128]` K = 8 vs dense, 20 Hz | +1.22 [+1.05, +1.66] | ×0.063 | ×1.00 | −0.31 |
| `[512,512]` K = 16 vs dense, 20 Hz | +1.85 [+1.38, +2.26] | ×0.031 | ×1.00 | −0.37 |
| `[512,512]` K = 16 vs `[128,128]` K = 8 | +0.64 [+0.22, +0.98] | ×7.99 | ×3.99 | +0.21 |
| `[128,128]` K = 8 at 20 Hz vs `[32,16,8]` at 40 Hz | +3.24 [+2.70, +3.89] | ×1.60 | ×3.44 | — |

"Equivalent" means TOST ±0.5 dB. The Wilcoxon p-values are in `pilot_b_contrasts.csv`; nothing
here is Holm-adjusted.

- **Past the architecture choices, fidelity costs about 0.2 dB per doubling of synaptic
  work.** Input width gives 0.17–0.23 dB and a wider fixed-fan-in layer 0.21 dB. Independent
  input channels would give 3 dB per doubling; this is the shared-error ceiling in cost units.
- **Fan-in is the only lever that raises the ceiling and cuts cost at once.** It gives
  +1.2 to +1.9 dB with 1/16–1/32 of the synaptic work.
- **Firing rates barely move synaptic work, because the encoder does most of it.** The
  encoder's 40 Hz spikes into layer 1 account for 80% of the synaptic operations in the
  rate-controlled `[32,16,8]`, 97% in `[256,16,8]` and 100% in any one-layer network. Output
  spikes feed only the readout. So the output rate, the largest operating-point lever R3
  found, is nearly free for synaptic work and saves at most 14% of spikes (0.25 Hz against 40 Hz).

### 1.4 Reading rule

The rule was written in the script before it first ran. Ceilings were already published in R3
and R4b, so this is a framing judgement, not a pre-registered test.

1. **A front point within 0.5 dB of the best ceiling at ≤ ¼ of its synaptic
   operations: not met.** `[128,128]` with K = 8 uses ⅛ of the best point's synaptic
   operations. Its paired gap to the best is +0.64 dB [+0.22, +0.98], outside the margin.
   **A correction on record:** the script's first run measured the gap as a difference of
   medians (0.43 dB) and reported the criterion as met. Every R3 comparison pairs by record, so
   the rule now says so, and the paired distance reverses that verdict.
2. **At least two levers on the front: met, but weakly.** Both levers on the front, removing a
   layer and fixing fan-in, yield one-layer networks.

**Verdict: inconclusive as a frame.** On the existing grids, adding cost to ICEE's fan-in rule
yields three statements, not a frame:

- the knee sits at `[128,128]` with K = 8;
- a second layer costs twice;
- past the knee, fidelity costs about 0.2 dB per doubling of synaptic work.

Grid B2 settles this in §3. The grids were built to test mechanism, not cost. They sample the front at three points, and
they never vary the two quantities that should set cost: the encoder rate, which drives 80–100%
of the synaptic work, and the width of the input pool behind a fixed fan-in.

### 1.5 What pilot B cannot see

- **The encoder rate.** Every rate-controlled grid holds the encoder at 40 Hz.
- **The input pool apart from layer width.** Every fixed-fan-in network is `[N,N]`. At fixed K
  and layer width, synaptic work is `r_0 · K · N_1`, independent of `N_in`, while M4's column
  cosine `3K/4N_in` falls as `N_in` grows. So ICEE's own shared-error model predicts a lever that
  costs encoder spikes but no synaptic work, and no grid has tested it.
- **The gap between 10⁴ and 4 × 10⁴ synaptic operations per second**, where small one-layer
  fixed-fan-in networks would sit. M4's `[32,32]` K sweep ran under the pre-R3 protocol and
  records no encoder rate.
- **Trained weights driven to zero.** They are counted as synapses, so trained structural counts
  are upper bounds.

## 2. Grid B2 — design

B2 is the grid pilot B needed to decide the frame. It needed no new library code, because
`r3_common` already builds a fixed fan-in for any `[N_in, N_1]` and takes
`Spec.encoder_rate_hz`. The predictions and decision rules are in `b2_grid.py`'s docstring,
written before the grid ran.

- **Design.** One LIF layer held at 20 Hz; K = 8; `N_1` ∈ {32, 128}; `N_in` ∈ {32, 128, 512,
  2048}; encoder rate ∈ {10, 20, 40} Hz; noiseless input; 16 records × 5 seeds. That is 24
  design points and 1,920 runs.
- **Cost.** At a fixed fan-in the synaptic work is `r_0 · K · N_1`, the same at every `N_in`.
  The input pool costs encoder spikes (`N_in · r_0` events) and nothing else.
- **P1.** At fixed `N_1` and K, the ceiling rises with `N_in`.
- **P2a.** The ceiling falls as the encoder rate drops.
- **P2b.** The cascade law with a Poisson encoder term, `1/R_out = c/r_0 + 1/Λ_rest` in `S/N`,
  is fitted at 20 and 40 Hz and predicts 10 Hz within 0.3 dB median absolute error.
  **Amended while the grid ran, before any row was analysed.** The first version backed the
  layer's intrinsic `S/N` out of the encoder stage's read-out. Three smoke-test runs showed the
  output SNR above that read-out's, which leaves the intrinsic undefined; the docstring gives
  the details.
- **F1.** `N_in` 2048 minus 128, at `N_1` = 128 and a 40 Hz encoder, is at least +0.5 dB,
  paired, with p < 0.05.
- **F2.** A B2 point dominates `[128,128]`, K = 8, at a 40 Hz encoder. The candidates were fixed
  by nominal cost before any ceiling was read.
- **Frame.** Cost against fidelity is the frame if F1 or F2 holds.

## 3. Grid B2 — results

**Check.** The `[128,128]`, 40 Hz cell repeats R4b's fan-in runs with the same random streams,
and all 80 runs reproduce R4b's ceilings exactly.

Ceiling (dB, median over records):

| `N_1` | encoder (Hz) | `N_in` 32 | 128 | 512 | 2048 | synaptic ops/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 40 | 3.67 | 4.36 | 4.72 | 4.86 | 10,240 |
| 32 | 20 | 3.00 | 3.97 | 4.21 | 4.29 | 5,120 |
| 32 | 10 | 2.27 | 3.25 | 3.75 | 3.86 | 2,560 |
| 128 | 40 | 3.97 | 5.15 | 5.55 | 5.68 | 40,960 |
| 128 | 20 | 3.18 | 4.47 | 5.22 | 5.41 | 20,480 |
| 128 | 10 | 2.51 | 3.90 | 4.66 | 5.11 | 10,240 |

### 3.1 Pre-registered tests

| test | result | verdict |
|---|---|---|
| P1, `N_1` = 32 | +0.20 [+0.12, +0.29] dB per doubling of `N_in`, p_holm 6 × 10⁻⁵ | **supported** |
| P1, `N_1` = 128 | +0.29 [+0.15, +0.42] dB per doubling of `N_in`, p_holm 6 × 10⁻⁵ | **supported** |
| F1 | +0.57 [+0.28, +0.90] dB, p 3 × 10⁻⁵ | **met** |
| P2a | 40 Hz minus 10 Hz: +0.67 to +1.59 dB, all 8 architectures p_holm 2.4 × 10⁻⁴ | **supported** |
| P2b | median \|error\| 0.067 dB (signed +0.05, 90th percentile 0.20) over 128 cells, none unbounded | **supported** |
| F2 | dominance over `[128,128]` at 40 Hz: 2 of 3 superiority and 5 of 20 non-inferiority candidates | **met** |

- **The input pool is a real lever.** Widening the pool behind a fixed fan-in raises the ceiling
  by 0.2–0.3 dB per doubling without changing the synaptic work. This is M4's decorrelation
  mechanism with fan-in held still, which no earlier grid could separate.
- **F1 clears its bar on the median but not on the interval.** The lower bound, +0.28, is
  below +0.5.
- **The encoder rate costs 0.34–0.80 dB per halving, and a wide pool makes it cheap.** The
  cost is 0.80 dB per halving at `[32,128]` and 0.34 at `[2048,128]`.
- **The 1/r_0 law prices the encoder rate** to 0.07 dB median error. It is a two-parameter fit
  extrapolated one halving, so it shows the form holds; it is not a zero-parameter prediction
  like R3's.
- **F2 superiority, at the same synaptic work:** `[2048,128]` at +0.57 [+0.28, +0.90] dB and
  `[512,128]` at +0.45 [+0.22, +0.66].
- **F2 non-inferiority:**
  - at ½ the synaptic work: `[512,128]` with a 20 Hz encoder, +0.07 [+0.04, +0.11];
    `[2048,128]` at 20 Hz, +0.24 [+0.16, +0.41];
  - at ¼: `[2048,128]` at 10 Hz, −0.07 [−0.13, +0.03]; `[512,32]` at 40 Hz, −0.26 [−0.35,
    −0.11]; `[2048,32]` at 40 Hz, −0.23 [−0.29, −0.11].

  These are 5 of 20 candidates; the intervals are not adjusted for multiplicity, but every
  lower bound clears −0.5 dB by at least 0.15.

**Frame decision: met, by F1 and by F2.**

### 3.2 The fronts, R3 + R4b + B2

| synaptic-operation front | ceiling (dB) | synaptic ops/s | spikes/s |
|---|---:|---:|---:|
| `[2048,32]`, encoder 10 Hz | 3.86 | 2,560 | 21,113 |
| `[2048,32]`, encoder 20 Hz | 4.29 | 5,117 | 41,569 |
| `[2048,128]`, encoder 10 Hz | 5.11 | 10,238 | 23,037 |
| `[2048,128]`, encoder 20 Hz | 5.41 | 20,466 | 43,485 |
| `[2048,128]`, encoder 40 Hz | 5.68 | 40,932 | 84,449 |
| `[512,512]`, K = 16, 20 Hz (R4b) | 5.77 | 327,500 | 30,713 |

The spike front has ten points:
- `[32,32]` at every encoder rate;
- `[128,32]` and `[128,128]` with a 20 Hz encoder;
- R4b's `[128,128]` with K = 8 at 10 and 20 Hz;
- `[512,128]` with a 40 Hz encoder, at 5.55 dB and 22,988 spikes/s;
- R4b's `[512,512]` with K = 16 at 10 and 20 Hz.

- **`[2048,128]` with a 10 Hz encoder reaches 5.11 dB on 10⁴ synaptic operations per
  second.** That is the synaptic work of `[32,8]` (2.75 dB), and 40% of the shipped `[32,16,8]`'s (2.6 × 10⁴
  for 1.92 dB).
- **Post hoc, `[512,128]` with K = 8 and a 40 Hz encoder matches the best ceiling measured
  before B2 and costs less on every axis.** Against R4b's `[512,512]`, K = 16, it scores
  −0.20 [−0.40, +0.02] dB, equivalent at ±0.5. It needs ⅛ of the synaptic operations, ⅛ of the
  synapses (1,024 against 8,192) and 0.75× the spikes. `[2048,128]` at 40 Hz is also
  equivalent (−0.03 [−0.25, +0.13]) at ⅛ of the synaptic work, but with 2.75× the spikes.
  Read after the fact, pilot B's criterion 1 is met by these points.
- **Pool width moves cost from synaptic operations to spikes.** Along the synaptic-operation
  front, spikes grow as `N_in · r_0`. The two fronts share only R4b's `[512,512]`; its
  `[512,128]` rival sits on the spike front. Which design is cheaper depends on a chip's
  relative cost per synaptic operation and per routed spike, which this project has not
  measured and must not assume. The paper reports both axes.

### 3.3 Post hoc: how much of the ceiling is encoder counting noise

This analysis was designed after the tests were read. It uses the fitted `1/R_out = c/r_0 +
1/Λ_rest` at the 40 Hz encoder.

| `[N_in, N_1]` | encoder share of output noise | `Λ_rest` (dB, affine) |
|---|---:|---:|
| `[32,32]` | 0.44 | 5.20 |
| `[128,32]` | 0.30 | 5.63 |
| `[512,32]` | 0.21 | 5.60 |
| `[2048,32]` | 0.19 | 5.55 |
| `[32,128]` | 0.44 | 5.60 |
| `[128,128]` | 0.22 | 5.82 |
| `[512,128]` | 0.15 | 5.94 |
| `[2048,128]` | 0.10 | 6.07 |

Encoder counting noise is 44% of the output noise with a 32-channel pool and 10% at 2048.
What it leaves, `Λ_rest`, barely moves: 5.6 to 6.1 dB at `N_1` = 128. That remainder is the
layer's own shared error. Neither encoder rate nor pool width can lift a K = 8 layer much past
about 6 dB on MIT-BIH, which is the flat ceiling again, now split into a part the input budget
buys down and a part it cannot.

### 3.4 What B2 does not show

- **The pool is tiled.** It is `N_in` independent Poisson draws of one ECG channel. Its cost is
  real address-event traffic, not free information.
- **Address width is not counted.** Indexing into 2048 inputs takes 11 bits per synapse
  against 7 for 128.
- **Scope.** One layer, K = 8, the layer held at 20 Hz, MIT-BIH only, 5 seeds, noiseless
  ceilings. R3 found ceilings flat in input SNR at every configuration it tested; B2 did not
  re-test that.
- **Energy.** No per-event energy figures. Pilot B's cost units are counts.

## 4. What H1 does to the paper

- **The frame.** Fidelity against hardware cost, as a noise budget the cascade law prices. Each
  term is new relative to ICEE:
  - encoder counting noise, which scales as 1/r_0;
  - input-pool overlap, which a wider pool removes at no synaptic cost;
  - the layer's own shared error, which caps the rest.
- **The headline result.** A single layer of 128 neurons with 8 inputs each, drawn from a
  512-channel pool, matches the best ceiling measured on MIT-BIH with ⅛ of the synaptic
  operations and synapses and fewer spikes. Report it as post hoc, or confirm it in the final
  grid with a pre-registered test.
- **What ICEE contributes** becomes cited background: the flat ceiling, the law and the fan-in
  rule.
- **Pilot A** joins the same budget. Weight quantization and device variation are further
  terms, and whether the law prices them too is the natural next test.

## 5. Pilot A — design (not run)

The rules are in the docstring of `scripts/h1_hardware_cost/pilot_a_grid.py`. The committed
numeric predictions are in `results/h1_hardware_cost/analysis/pilot_a_prediction{s,_summary}.csv`,
written by `pilot_a_predictions.py` before any grid row exists.

**Question.** B2 split the output noise into an encoder counting term and a remainder. A neuron
averages counting noise over its effective fan-in, `PR = (Σw)²/Σw²`, which R4b found tracks a
layer's noise. Weight quantization, lognormal device-to-device variation and open
(stuck-at-zero) devices all lower PR by amounts computable from the weight matrices alone. If the
budget is right, each one raises the counting term by `PR_exact / PR_level` and leaves the
remainder untouched: a zero-parameter prediction.

**Design.**
- **Networks:** the three `N_1` = 128 networks from B2, `[128,128]`, `[512,128]` and
  `[2048,128]`, with K = 8 and the layer at 20 Hz.
- **Encoder:** 10, 20 or 40 Hz.
- **Device levels, one non-ideality at a time:** exact; 1, 2, 3 or 4 bits; σ = 0.25, 0.5, 1.0
  or 1.5; 25%, 50% or 75% of devices stuck at zero.
- **Size:** 16 records × 5 seeds, 108 design points, 8,640 runs, about 30 min.
- **Code:** `Spec` gains `weight_bits`, `weight_sigma` and `weight_stuck`, applied to every
  weight matrix before calibration by `r3_common.perturb_weights`. The default path is
  unchanged: B2's `[128,128]` run still gives 3.292972653768322 dB, and existing spec keys do
  not change.

| test | prediction |
|---|---|
| PA1, precision budget | the fewest bits equivalent to exact weights (TOST ±0.5 dB) is ≤ 2 at all 9 network × rate points |
| PA2, variation budget | the largest σ equivalent to exact is ≥ 0.5 at all 9 |
| PA3, the budget prices the faults | median \|error\| ≤ 0.3 dB over all 99 perturbed points × 16 records, **and** better than a no-change model on the 22 points predicted to lose ≥ 0.5 dB |
| PA4, is PR the whole story | the matched-PR pairs 50% stuck vs σ 1.5 and 1 bit vs σ 1.0, whose spread of summed weight differs 2–3×, are equivalent at all 9 points |

The largest predicted losses, as medians over records:
- 75% stuck: −1.19 to −1.53 dB at `[128,128]` and −0.61 to −1.34 dB at `[2048,128]`;
- 50% stuck and σ 1.5: −0.57 to −0.79 dB at `[128,128]`;
- 1 bit: −0.16 to −0.51 dB;
- 2 bits and σ ≤ 0.5: ≤ 0.17 dB.

The model predicts that a wide pool and a fast encoder protect against device faults, because
both shrink the counting term that faults inflate.

**Reading rule.**
- **PA3 supported:** the paper's budget prices encoder rate, input pool, weight precision and
  device faults with one model.
- **PA3 not supported:** PA1 and PA2 still give an empirical precision and fault table, and the
  model claim stays with the encoder.
- **Either way:** weight memory (bits × synapses) becomes a cost axis.

**Disclosure.** After the levels and predictions were fixed, the code was smoke-tested once per
non-ideality (record 100, seed 0, `[128,128]`, 40 Hz). All three lost less than the model
predicts: −0.07 dB at 1 bit, −0.13 at σ 1.5 and −0.06 at 50% stuck, against medians of −0.34,
−0.57 and −0.57. These are three runs out of 8,640 and nothing was changed after them, but they
point to PA3's second condition as the one at risk.

**Not covered.**
- **Cycle-to-cycle read noise:** a fresh weight error at every synaptic event needs a change to
  the Numba synapse kernel.
- **Event loss in the interconnect:** dropping each input spike with probability p is Poisson
  thinning, which B2's encoder-rate cells already price as a rate of `r_0(1 − p)`.
- **Faults and quantization together.**

