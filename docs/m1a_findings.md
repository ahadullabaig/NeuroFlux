# M1a — the shared-input control

**Status:** resolved. Pre-registered interpretation: **"ceiling holds."**
**Data:** `results/m1a_shared_input/m1a_results.csv` (54 rows, 0 degenerate).
**Reproduce:** `uv run python scripts/m1a_shared_input/run_m1a.py --seeds 3 --duration 8.0`

---

## 1. Why this experiment existed

R0 found that the Finding 4 capacity sweep could not test what it claimed to.
`cli.py:289` builds the network's drive as `np.tile(noisy, (n_input, 1)).T` — every
input channel carries the *same* noisy scalar. The information the input holds about the
clean signal is therefore one sample's worth regardless of `n_input`.

Moreno-Bote et al. (2014, *Nat. Neurosci.* 17:1410) prove that Fisher information **must**
saturate under exactly this condition:

> "If the information provided by the outside world about a stimulus, s, is finite, then
> information must saturate as we increase the number of neurons coding for s."

and identify the mechanism:

> "If neurons share input, the resulting shared variability in their response cannot be
> averaged out."

So the measured flatness had a mundane candidate explanation: not an architectural
ceiling, but an artifact of how the input was constructed. If true, the paper's central
claim collapses into a design error. M1a distinguishes the two.

## 2. Design

Three input modes, identical in every other respect:

| Mode | Construction | What it tests |
|---|---|---|
| `tiled` | `np.tile(noisy, (n_input, 1)).T` | Current behaviour. Control. |
| `independent` | `n_input` independent noise realisations of the same clean signal | **The hypothesis test.** Information genuinely grows with width; an ideal estimator gains `10·log₁₀(N)` dB, i.e. **+3.01 dB per doubling**. |
| `delay` | Channel *k* carries `noisy` lagged by *k* samples | Realistic engineering variant. Adds temporal context, not repeated measurements, so the +3 dB reference does not apply to it. |

Every condition runs under **per-layer rate calibration** to 40 Hz (`calibrate_gain`), so
width is never confounded with operating point. Every reconstruction is gated on
`is_degenerate` before scoring, and scored with `affine_corrected_snr`. Passive network
(`epochs=0`), synthetic ECG, 0 dB input SNR, 3 seeds, 8 s, no flicker.

## 3. Result — the ceiling holds

**Zero of 54 runs were degenerate**, and `out_std` stayed in 0.235–0.411 throughout, so no
result below is a silent or saturated network.

### 3.1 The decisive column: `independent`

| N_in | ceiling (dB) | ideal (dB) | **shortfall** |
|---:|---:|---:|---:|
| 8 | 0.31 ± 0.02 | 9.03 | 8.73 |
| 16 | 0.46 ± 0.08 | 12.04 | 11.59 |
| 32 | 0.41 ± 0.06 | 15.05 | 14.64 |
| 64 | 0.57 ± 0.04 | 18.06 | 17.49 |
| 128 | 0.63 ± 0.02 | 21.07 | 20.44 |
| 256 | 0.56 ± 0.03 | 24.08 | 23.52 |

**Slope: +0.056 dB per doubling, against an ideal of +3.01.** Over a 32× increase in input
width the network gained **+0.25 dB** where an ideal estimator gains **+15 dB**. The
shortfall does not merely persist as width grows — it *widens monotonically*, from 8.7 dB
to 23.5 dB.

Giving the network genuinely independent information about the clean signal changes
essentially nothing.

### 3.2 The other two modes

| Mode | slope (dB/doubling) | total 8 → 256 |
|---|---:|---:|
| `tiled` | +0.090 | +0.40 dB |
| `independent` | +0.056 | +0.25 dB |
| `delay` | −0.029 | −0.17 dB |

`delay` collapses to 0.00 dB by N=64 — lagged channels destroy the instantaneous rate code
the population decoder reads, which is expected and is a useful negative control rather
than a surprise.

Note that `tiled` and `independent` are statistically indistinguishable. That is the
finding: the input mode does not matter.

## 4. The ceiling does not move with input quality either

A second control, flicker-free, at N_in = 32 (`scratchpad/m1a_snr_axis.py`):

| input SNR (dB) | `tiled` ceiling | `independent` ceiling |
|---:|---:|---:|
| 0 | 0.51 ± 0.08 | 0.41 ± 0.05 |
| 5 | 0.96 ± 0.10 | 0.70 ± 0.07 |
| 10 | 1.10 ± 0.06 | 0.92 ± 0.03 |
| 20 | 1.33 ± 0.13 | 1.26 ± 0.08 |
| 40 | 1.44 ± 0.10 | 1.47 ± 0.12 |

**~1 dB of output across a 40 dB range of input** — a slope of roughly 0.025 dB per dB.

Two things follow. First, **Finding 1 survives the flicker control**: the flatness is not
an artifact of the input-SNR axis saturating near −6 dB, because this pipeline applies no
flicker at all and the axis spans its full nominal range. Second, the 40 dB column is
effectively a clean-signal measurement, and it lands at **1.44–1.47 dB** — consistent with
`phase3_findings.md` §5.6.1, which measured intrinsic output SNR ≈ 0.7 dB on a clean
signal by a different route. Two independent measurements agree that the architecture's
intrinsic reconstruction ceiling is on the order of 1 dB.

## 5. What this establishes, and what it does not

**Established.** The ceiling is not an artifact of the tiled input. The shared-input /
information-limiting-correlations explanation is **falsified** for this system: the
bottleneck lies downstream of the input representation. It is also not a gain-calibration
artifact (every layer held 40 Hz), not a degeneracy artifact (0/54 degenerate), and not a
flicker artifact (no flicker applied).

**Not established.** *Where* the information dies. The candidates are the Poisson encoder,
the LIF layer, and the population decoder, and this experiment does not separate them.
That is M2, which is now the centrepiece of the paper rather than a supporting result.

**Caveats, stated plainly.**
- Passive network only (`epochs=0`). Finding 2 measured training as worth 0.11–0.44 dB, so
  this is unlikely to change the conclusion, but it has not been re-checked under per-layer
  rate control.
- Synthetic ECG, 3 seeds. Real MIT-BIH confirmation and proper power are still owed (R4).
- The absolute ceiling (~0.5–1.5 dB affine-corrected) is low enough that the reconstruction
  is poor in absolute terms. "Invariant ceiling" and "barely works at all" are not the same
  claim, and the manuscript must not blur them. The input-SNR sweep in §4 is what
  distinguishes them: output *does* respond to input quality, just far too weakly.

## 6. Consequence for the paper

The pre-registered reading of this outcome (roadmap R4/M1a) was: *"the shared-input
explanation is falsified, something downstream binds, and the paper is substantially more
interesting."* That is the branch we are on.

The contribution is now: a rate-coded unsupervised-STDP reconstruction pipeline exhibits a
~1 dB output ceiling that is invariant to input information (32× width, 15 dB of available
gain forgone), input quality (40 dB range), output width, decoder time constant, training
duration, encoder and decoder — and the cause is internal to the spiking representation,
not the input coding. M2 must localise it.

---

## Real MIT-BIH — the ceiling holds on patient ECG

**Data:** `results/m1a_shared_input/m1a_mitbih.csv` — 16 records × 3 input modes × 6 widths ×
10 seeds, 2,880 rows, **0 degenerate**. Each seed takes a different artifact-free window of its
record. **Reproduce:** `uv run python scripts/m1a_shared_input/run_m1a.py --dataset mit-bih --seeds 10`.
The script was refactored to take real records first, and verified to reproduce
`m1a_results.csv` byte for byte before this run.

The record is the unit of analysis: a slope is fitted per record, then summarised across the 16.

| Mode | slope, dB per doubling (mean ± sd across records) | range across records | ceiling at N = 256 |
|---|---:|---:|---:|
| `tiled` | +0.102 ± 0.028 | [+0.046, +0.150] | 0.92 ± 0.27 dB |
| `independent` | **+0.086 ± 0.023** | [+0.051, +0.133] | 0.76 ± 0.21 dB |
| `delay` | −0.044 ± 0.031 | [−0.121, −0.005] | 0.03 ± 0.04 dB |

**The pre-registered verdict replicates.** Under `independent` input — where information really
does grow with width and an ideal estimator gains +3.01 dB per doubling — the network gains
+0.086 dB per doubling, **2.8% of the ideal**. The largest slope on any of the 16 records is
+0.133. One-sided Wilcoxon against 3.01: p = 1.5 × 10⁻⁵.

**Giving the network more information makes it very slightly worse, not better.** Paired on
record, `independent` minus `tiled` is −0.017 dB per doubling (range −0.030 to +0.005,
Wilcoxon p = 0.0001). The same ordering appeared on synthetic ECG (+0.056 against +0.090). The
effect is far too small to matter — a tenth of a dB across the full 32× range — but its direction
rules out the reading M1a existed to test: if the tiled construction were what capped the
ceiling, independent input would lift it.

`delay` again collapses toward zero, as on synthetic: lagged channels smear the instantaneous
rate code the population decoder reads.

**What this closes.** The shared-input objection is now refuted on synthetic *and* real ECG, with
the record as the unit of analysis and 10 seeds. M3's cascade law gives the reason the input mode
cannot matter: the binding losses are the stages' intrinsic `1/Λ` terms, which depend on the
weight columns and the spiking noise, not on how much information arrives.
