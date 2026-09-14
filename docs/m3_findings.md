# M3 — the analytical bound

**Status:** resolved. The ceiling is derived, predicted with zero free parameters, and
traced to a single number that is fixed before the first spike is simulated.
**Data:** `results/m3_bound/`
**Library:** `packages/core/src/neuroflux_core/metrics/cascade.py`, tested in
`tests/core/metrics/test_cascade.py` (36 tests).
**Reproduce:**
```
uv run python scripts/m3_bound/run_m3.py --seeds 3 --layers 32 16 8 --tau 20
uv run python scripts/m3_bound/run_m3.py --seeds 3 --layers 128 128 128 --tau 4
uv run python scripts/m3_bound/noise_structure.py --seeds 3
uv run python scripts/m3_bound/weight_correlation.py --seeds 3 --width 512
```

> **Correction (R3, `docs/r3_findings.md` §1.3) — read before §3 and §8.1.** The law is
> derived for `S/N`, but every number here applies it to *affine-corrected* SNR, which equals
> `1 + S/N` and cannot fall below 0 dB. That is why the law "under-predicts by 0.8–2.8 dB"
> below 10 dB input. Applied in `S/N` terms (`affine_to_excess_db`, now in
> `metrics/cascade.py`) the low-SNR residual is −0.3 dB, not +1.6 dB: the network does
> slightly *worse* than the law there, not better. §3's limiter explanation is also wrong on
> its own terms — `v_min`/`v_max` are the drive's own extremes, so the `[0, 1]` clip never
> engages. **"The Phase 3 regime advantage, derived" is withdrawn.** The ≥ 10 dB validation,
> the ceiling values, §4–§6 and §8.2–§8.6 are unaffected: the two forms agree to within
> 0.1 dB once every term is well above 0 dB. Two further R3 qualifications: every number here
> is at the 40 Hz operating point, and the ECG ceiling is higher at a sparser output
> (`r3_findings.md` §3.2); and §4's column cosine predicts a stage's noise for *random*
> weights but not after STDP, which decorrelates layer 1 while lowering its intrinsic SNR (§2.6).

---

## 1. A correction to M2 first

M2 reported that *"each LIF layer retains roughly 22% of the information reaching it"*,
citing 8.56 dB → 1.90 dB and 1.90 dB → 0.46 dB. Those ratios were computed **on the
decibel values**, not on the powers they represent:

| stage | in (dB) | out (dB) | ratio of dB | ratio of powers |
|---|---:|---:|---:|---:|
| layer1 | 8.56 | 1.90 | 22.2% | 21.6% |
| layer2 | 1.90 | 0.46 | 24.2% | **71.8%** |
| layer3 | 0.46 | 0.37 | 80.4% | **97.9%** |

A ratio of logarithms is not a meaningful quantity — it changes if the reference level
changes. Layer 1's agreement (22.2 vs 21.6) is a coincidence of where those two numbers
happen to sit. **The apparent constancy across layers was an artifact, and the "~22% per
layer" figure should not appear in the manuscript.**

This matters beyond bookkeeping: it was the *constancy* that made "22%" look like a law
worth deriving. There is a law here, but it is not that one, and §2 gives it.

## 2. The law: inverse SNRs add

Write stage `k`'s population readout as what it inherits plus what it adds:

```
y_k = G_k * y_{k-1} + eta_k
```

`G_k` is the stage's transfer gain and `eta_k` its own spike-sampling noise, independent
of everything upstream. Signal power scales as `G_k^2 S_{k-1}` and noise power as
`G_k^2 N_{k-1} + var(eta_k)`, so the gain cancels from the ratio:

```
1/SNR_k = 1/SNR_{k-1} + var(eta_k)/(G_k^2 S_{k-1})
        = 1/SNR_{k-1} + 1/Lambda_k
```

`Lambda_k` is the stage's **intrinsic SNR** — what it would deliver from a noiseless
input. Telescoping along the pipeline:

> **1/SNR_out = 1/SNR_in + sum_k 1/Lambda_k**

This is Friis's noise-figure formula for a cascade of amplifiers, and it has the same
algebraic form as the Fisher-information saturation of Moreno-Bote et al. (2014),
`I_N = I_0/(1 + eps*I_0)`. Unlike retention, `Lambda_k` does not depend on what arrives,
so it is the quantity that can be compared across stages and configurations.

Three consequences, each of which the project had already measured without explaining:

1. **The ceiling is the clean-signal output.** As `SNR_in -> inf` the first term vanishes
   and `SNR_out -> 1/sum_k(1/Lambda_k)`, a finite constant set entirely by the
   architecture. **Finding 1's flatness is this limit.**
2. **Depth is strictly harmful.** Every stage contributes a positive term.
3. **Only the smallest `Lambda` matters.** Improving a stage whose `1/Lambda_k` is a small
   part of the sum moves nothing. **Findings 2–5 and M1a are all measurements of
   non-binding stages.**

## 3. The test: predicting noisy runs from a clean run, with nothing fitted

Run the pipeline on a **noiseless** signal. With `1/SNR_in = 0` the measured per-stage
SNRs *are* the `Lambda_k`. Then predict the output at every other input SNR. No parameter
is fitted to the noisy runs.

**Shipped `[32,16,8]`, tau=20 ms.** Clean-input intrinsics: encoder 3.87, layer1 8.59,
layer2 7.76 dB → predicted ceiling **1.45 dB**.

| SNR_in (dB) | stage | predicted | measured | residual |
|---:|---|---:|---:|---:|
| 9.87 | layer2 | 0.87 | 1.10 | +0.24 |
| 19.47 | encoded | 3.75 | 3.74 | **−0.01** |
| 19.47 | layer1 | 2.52 | 2.48 | **−0.04** |
| 19.47 | layer2 | 1.38 | 1.33 | **−0.05** |
| 29.43 | layer2 | 1.44 | 1.42 | −0.02 |
| 39.43 | layer2 | 1.45 | 1.44 | −0.01 |

**Wide `[128,128,128]`, tau=4 ms.** Intrinsics: encoder 9.12, layer1 3.13, layer2 5.72 dB
→ predicted ceiling **0.57 dB**; measured at 39.43 dB input, **0.54 dB**.

**Best `[512,512]`, tau=20 ms.** Intrinsics: encoder 4.68, layer1 11.58 dB → predicted
ceiling **3.87 dB**; measured **3.85 dB**.

Across all three configurations, **at input SNR ≥ 10 dB the prediction is accurate to
within 0.2 dB with zero free parameters.** The ceiling is not an empirical regularity; it
is `1/sum_k(1/Lambda_k)`, and it can be measured in a single clean-input run.

### Where the law fails, and why that is the good news

> **Withdrawn — see the correction at the top.** The deviation below is the `1 + S/N` floor
> of the affine metric, not a limiter; in `S/N` terms it is small and negative.

Below ~10 dB input the model **under**-predicts by 0.8–2.8 dB: the network does better
than the law says it should. That is a real effect and it is the one place the pipeline
adds value. The encoder's clip to `[0,1]` is a limiter, so large noise excursions are
compressed while the signal is not. The law assumes additive independent noise; a limiter
violates that assumption in the network's favour.

**This is the Phase 3 regime advantage, derived.** `phase3_findings.md` reports that the
SNN improves ΔSNR at input SNR ≤ 0 dB and degrades it above — and the deviation from the
cascade law is confined to exactly that regime, growing monotonically as input SNR falls.

## 4. Where `Lambda` comes from: shared error

Split each neuron's smoothed trace into the part a linear decoder can predict from the
clean signal and the part it cannot, `trace_i = a_i*clean + b_i + r_i`. Averaging N neurons
shrinks the *independent* part of `r_i` but not the part they share:

```
var_mean = var_single * (1/N + c*(1 - 1/N))
Lambda(N) = var_signal / var_mean       ->   var_signal/(c*var_single)  as N -> inf
```

One number, `c`, caps what any width can buy. Measured on a clean input, `[N,N]`,
tau=20 ms:

| N | Λ measured | Λ if c=0 | c | Λ at N→∞ | repeatable |
|---:|---:|---:|---:|---:|---:|
| 16 | −1.88 | 7.15 | 0.468 | −1.58 | 13.1% |
| 32 | −0.18 | 11.14 | 0.406 | 0.01 | 15.3% |
| 64 | 1.02 | 14.86 | 0.369 | 1.13 | 18.4% |
| 128 | 1.43 | 17.97 | 0.347 | 1.49 | 20.4% |
| 256 | 1.68 | 20.97 | 0.329 | 1.71 | 22.8% |
| 512 | **1.90** | **24.15** | 0.327 | **1.92** | 25.2% |

Three readings:

- **The cost of correlation is 22 dB.** At N=512 the layer would deliver 24.15 dB if its
  neurons erred independently. It delivers 1.90.
- **The population is already saturated.** Measured Λ tracks the `N→∞` limit to within
  0.03 dB from N=64 upward. Adding neurons buys ~0.2 dB per doubling against the +3.01 dB
  that independence would give. **This is M1a's +0.056 dB/doubling and Finding 2's
  +0.16 dB over 16×, derived rather than observed.**
- **The shared error is mostly stochastic, not distortion.** Re-running the same network
  on the same analog drive with a different encoder RNG leaves only 13–25% of each
  neuron's error repeatable. So this is shared *noise*, not the LIF transfer nonlinearity.

The encoder, by contrast, has `c ≈ 0.07` — an order of magnitude lower. The correlation is
manufactured by the LIF layer, which is why M2 localised the loss there.

## 5. The source: non-negative weights

Neuron `i`'s drive is `d_i = sum_k W[k,i]*s_k`. For roughly independent inputs the
correlation between two neurons' drives is **the cosine similarity of their weight
columns**, and for iid entries with mean `mu` and variance `sigma^2` that is
`mu^2/(mu^2 + sigma^2)`.

Every initialisation scheme in `network/topology.py` draws from a **non-negative** range,
because STDP bounds weights to `[w_min, w_max] = [0, 1]`. For any uniform draw on `[0,a]`,
`mu = a/2` and `sigma^2 = a^2/12`, giving

> **cos → (a/2)² / ((a/2)² + a²/12) = 3/4, independently of `a`.**

Measured on the actual sampled matrices, no simulation involved:

| scheme | mean | std | cos predicted | cos measured |
|---|---:|---:|---:|---:|
| uniform `[0,1]` | 0.5017 | 0.2882 | 0.7519 | **0.7525** |
| scaled `[0,0.3]` | 0.1505 | 0.0865 | 0.7519 | **0.7525** |
| gaussian | 0.5010 | 0.1656 | 0.9015 | 0.9022 |
| sparse (ρ=0.1) | 0.0502 | 0.1751 | 0.0759 | 0.0764 |

The prediction is exact to four decimals, and `uniform` and `scaled` — a 3.3× difference
in weight scale — give **identical** correlation. The scale cancels.

**This is why gain calibration could not move the ceiling.** M1b's per-layer calibration
rescales weights; correlation is scale-invariant. The calibration fixed the degeneracy
(output std held at ~0.10 where global gain collapsed it to 0) and left the ceiling
untouched, which is exactly what the algebra requires.

## 6. The intervention, and what it is worth

If `c` is set by column cosine similarity, lowering it should raise the ceiling. Sparse
initialisation does exactly that, and it is **already in the library, tested, and unused**.
Clean input, `[N,N]`, tau=20 ms, all schemes rate-calibrated to 40 Hz:

| scheme | cos | err corr `c` | N=32 | N=128 | N=512 |
|---|---:|---:|---:|---:|---:|
| gaussian | 0.902 | 0.42–0.49 | 2.44 | 3.10 | 3.26 |
| uniform | 0.750 | 0.33–0.40 | 2.74 | 3.55 | 3.87 |
| scaled (shipped) | 0.750 | 0.33–0.41 | 2.73 | 3.55 | 3.87 |
| **sparse** | **0.075** | **0.11–0.19** | 2.77 | **4.46** | **5.00** |

Output SNR in dB. Three readings:

1. **The ranking is monotone in cosine similarity at every width** — three schemes, three
   widths, no inversions. Firing rate is held at 40.0–40.7 Hz throughout, so this is not
   an operating-point effect.
2. **The benefit grows with width, as the model requires.** At N=32 sparse is worth
   +0.04 dB; at N=512, **+1.13 dB**. Low `c` raises the `N→∞` ceiling, so width keeps
   paying for longer instead of saturating.
3. **`[512,512]` sparse reaches 5.00 dB against the shipped default's 1.33 dB** — +3.67 dB,
   the largest single improvement found anywhere in this project, from a one-word change
   to an argument that already exists.

**Caveat.** At ρ=0.1 and N=32 each neuron receives ~3 inputs, so part of sparse's
width-dependence is fan-in growth rather than decorrelation alone. Separating the two needs
a density sweep at fixed fan-in, which is not yet run. The N≥128 numbers are not affected
by this.

## 7. Consequences

**For the paper.** M3 delivers what R4 asked for and more than expected. The claim is now:

> A rate-coded spiking reconstruction pipeline obeys `1/SNR_out = 1/SNR_in + sum_k
> 1/Lambda_k`, so its output quality is bounded by `1/sum_k(1/Lambda_k)` regardless of
> input quality. Each `Lambda_k` is capped by the mean pairwise error correlation among
> that stage's neurons, which for any non-negative weight initialisation is fixed at
> `mu^2/(mu^2+sigma^2) = 3/4` before training begins and is invariant to synaptic scale.
> Decorrelating the weight columns lifts the ceiling by 3.67 dB.

That is a derivation, a zero-parameter prediction validated on three configurations, a
mechanism, and an actionable fix. It also absorbs Findings 1–5, M1a and M2 as special
cases, and it explains the Phase 3 regime advantage as the one place the model's
assumptions break. *(Withdrawn by R3 §1.3: that "explanation" was the affine metric's
0 dB floor.)*

**Section III now has real content**, and Fig. 5 (predicted vs measured) is a diagonal
line across three configurations and six input SNRs.

**What this does not yet establish.**
- Synthetic ECG only, 3 seeds, single 8 s record. Real MIT-BIH and ≥10 seeds still owed.
- `c` is measured, not derived, *after* the LIF nonlinearity — §5 derives the correlation
  of the *drive*, and the step from drive correlation to output-error correlation
  (0.75 → 0.33) is empirical. Deriving that step needs the LIF transfer function
  linearised about the calibrated operating point.
- The sparse intervention is measured at one density (ρ=0.1, the library default). A
  density sweep at fixed fan-in would separate decorrelation from fan-in.
- Mixed-sign weights would drive `mu → 0` and `cos → 0`, which the model says is the
  strongest available intervention. STDP's `[0,1]` bound forbids it, so testing it needs
  a signed-weight variant of `STDPSynapse` — not attempted here.

## 8. Real MIT-BIH — the law holds

Everything above is synthetic ECG. All 16 Phase 3 records are now cached and verified
against the downloader manifest, and the mechanism has been re-measured on them. **The
unit of analysis is the record**, per `docs/r2_findings.md`: each record gets its own
`Lambda_k` from its own clean run, and its own prediction is checked against its own
measurement. Sixteen records are sixteen independent tests of the law, not one pooled test.

```
uv run python scripts/m3_bound/run_m3.py --dataset mit-bih --seeds 2 \
    --layers 32 16 8 --tau 20 --input-snrs 0 10 20 40
```

### 8.1 The zero-parameter prediction

| SNR_in (dB) | residual, mean ± sd across records |
|---:|---|
| 3.00 | +1.58 ± 0.29 |
| 10.39 | +0.34 ± 0.16 |
| 20.02 | **+0.02 ± 0.06** |
| 39.97 | **−0.00 ± 0.02** |

| regime | n | mean | median | RMS | max abs |
|---|---:|---:|---:|---:|---:|
| input SNR ≥ 10 dB | 144 | +0.121 | +0.023 | **0.234** | 1.112 |
| input SNR < 10 dB | 48 | +1.575 | +1.505 | 1.613 | 2.662 |

**At 20 and 40 dB input the prediction is exact to within measurement noise on real
patient ECG, with nothing fitted.** The low-SNR deviation is the same sign and the same
rough magnitude as on synthetic (+0.8 to +2.8 dB), so §3's limiter explanation — and with
it the derivation of Phase 3's regime advantage — transfers to real data.

> **Withdrawn (R3 §1.3).** The +1.58 dB at 3 dB input is the affine metric's `1 + S/N` floor.
> In R3's re-measurement (C1, 16 records × 10 seeds, rate-controlled) the law applied in
> `S/N` terms leaves a residual below 10 dB of −0.30 dB mean, 0.39 dB RMS.

Measured ceiling across the 16 records: **1.71 ± 0.76 dB**, range [0.67, 3.55] (synthetic
gave 1.45 dB). Per-record intrinsics: encoder 2.45–6.87, layer1 8.48–15.88, layer2
6.79–9.95 dB. **The encoder is the binding stage on every one of the 16 records**, matching
the synthetic shipped configuration and confirming §2's point that which stage binds is a
property of the configuration rather than of the data.

### 8.2 The mechanism

> **Superseded by §8.6.1** (16 records × 10 seeds). The two-record `c` values below were
> unrepresentative: at full power `c` is 0.35–0.44, not 0.19–0.29, it barely drifts with width,
> and the claim that real-data `c` sits below synthetic — offered here as the reason the real
> ceiling is higher — does not survive. The 20+ dB independence gap does.

Error structure, records 214 and 231, `[N,N]`, tau=20 ms (`noise_structure.py --dataset mit-bih`):

| N | Λ measured | Λ if c=0 | c | Λ at N→∞ |
|---:|---:|---:|---:|---:|
| 32 | −1.09 | 8.87 | 0.289 | −0.76 |
| 128 | 1.60 | 15.71 | 0.198 | 1.75 |
| 512 | **2.36** | **22.04** | 0.186 | **2.40** |

**A 19.7 dB gap at N=512, entirely attributable to correlated error.** Measured Λ tracks
the `N→∞` limit to within 0.04 dB from N=128 upward, so the population is saturated and
width buys ~0.3 dB per doubling against the +3.01 dB independence would give. Real-data `c`
(0.19–0.37) sits below synthetic (0.33–0.47), which is why the real ceiling is slightly
higher.

### 8.3 The intervention

> **Restated at full power in §8.6.2** (16 records × 10 seeds). Every conclusion holds; the
> numbers move by a few tenths of a dB.

4 records × 2 seeds, all schemes rate-calibrated (39.0–39.5 Hz):

| scheme | cos | c | Λ (N=512) | SNR out (N=128) | SNR out (N=512) |
|---|---:|---:|---:|---:|---:|
| gaussian | 0.901 | 0.287 | 0.41 | 2.90 | 3.25 |
| uniform / scaled (shipped) | 0.750 | 0.216 | 1.69 | 3.45 | 3.98 |
| **sparse** | **0.075** | **0.115** | **3.50** | **4.13** | **5.14** |

The 3/4 cosine prediction is unchanged (0.750–0.751 measured) because it is pure weight
algebra and does not involve the data at all. The ranking is monotone in cosine similarity
at both widths, as on synthetic. **`[512,512]` sparse reaches 5.14 dB against the shipped
`[32,16,8]` default's 1.71 dB — +3.43 dB on real patient ECG**, against +3.67 dB synthetic.

### 8.4 M2's destruction-vs-decoder control, on real data

16 records, shipped configuration, 20 dB input (`run_m2.py --dataset mit-bih`):

| stage | population | linear (test) | clipped |
|---|---:|---:|---:|
| analog input | 20.05 ± 0.02 | 19.88 ± 0.30 | 0.00% |
| encoded | 3.79 ± 1.15 | 3.28 ± 1.18 | 6.54% |
| layer1 | 2.76 ± 0.98 | 2.84 ± 0.98 | 0.91% |
| layer2 | **1.65 ± 0.70** | 1.71 ± 0.73 | 0.71% |

Population and optimal linear readout agree at every stage, as on synthetic. **The
information is genuinely destroyed on real ECG too; a better decoder will not recover it.**
This discharges the roadmap's "re-run M2 on real MIT-BIH".

### 8.5 What real data changed, honestly

- ~~**The single-anchor extrapolation is weaker than on synthetic.**~~ **Corrected in §8.6.1.**
  At two records, predicting Λ from `c` at N=32 undershot by up to +2.04 dB and `c` appeared to
  drift 0.37 → 0.19. At 16 records × 10 seeds the drift is 0.40 → 0.36 and the undershoot is at
  most +1.04 dB (layer 1, N=256). The model is now stated in measured form (§8.6.1): `c` measured
  at each width, no extrapolation.
- ~~**Encoder firing rate varies far more on real records.**~~ **Controlled in §8.6.3.** The
  rate did vary (15.6–65.2 Hz at 10 seeds), but holding it at 40 Hz leaves the cross-record spread
  of encoder Λ essentially unchanged, and the encoder still binds on all 16 records.
- ~~**Power is still uneven.**~~ **Resolved in §8.6.** Every real-data mechanism result is now
  16 records × 10 seeds.

### 8.6 Full power: 16 records × 10 seeds, and the encoder control

Every §8 experiment re-run at the roadmap's power commitment. Sources:
`results/m3_bound/m3_cascade_mitbih_s10.csv`, `m3_cascade_mitbih_s10_encoder40.csv`,
`m3_noise_structure_mitbih_s10.csv`, `m3_weight_correlation_mitbih_w{128,512}_s10.csv`.

#### 8.6.1 The width model, in measured form

`noise_structure.py`, `[N,N]`, τ = 20 ms, clean input, every width measured directly:

| N | encoder Λ | encoder `c` | layer-1 Λ measured | layer-1 Λ if c = 0 | layer-1 `c` | layer-1 Λ at N→∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.13 | 0.055 | −1.93 | 6.80 | 0.444 | −1.57 |
| 32 | 1.58 | 0.053 | −0.19 | 10.82 | 0.396 | 0.06 |
| 64 | 2.58 | 0.053 | 1.00 | 14.50 | 0.369 | 1.15 |
| 128 | 3.23 | 0.053 | 1.73 | 17.89 | 0.353 | 1.82 |
| 256 | 3.56 | 0.053 | 2.05 | 21.10 | 0.350 | 2.09 |
| 512 | 3.78 | 0.053 | **2.01** | **24.23** | 0.362 | **2.03** |

**The ceiling is now visible in the measurements themselves.** Layer-1 Λ plateaus at 2.0 dB from
N = 256: doubling width to 512 changes it by −0.04 dB, against the 22.2 dB gap to what independent
neurons would deliver. No extrapolation is involved.

Stated this way, the width model is: `Λ(N) = var_signal / (var_single · (1/N + c_N(1 − 1/N)))`
with `c_N` measured at each width. At the widths that matter (N ≥ 128), `c_N` is flat at 0.35–0.36,
so `Λ(N)` is within 0.1 dB of its `N → ∞` limit, `var_signal / (c · var_single)`. Extrapolating from
a single anchor at N = 32 still undershoots — by up to +1.04 dB for layer 1 — because `c` falls
from 0.40 to 0.35 between N = 32 and 128; the manuscript should quote measured Λ, not predicted.

The encoder behaves as the model's idealisation: `c` is constant at 0.053–0.055 across a 32× range of
width, and its extrapolation error is +0.6 dB at most.

Real-data layer-1 `c` (0.35–0.44) is **not** lower than synthetic (0.33–0.47), so §8.2's
explanation for the slightly higher real ceiling is withdrawn; the difference sits in `var_signal`
and the encoder stage, not in the correlation.

#### 8.6.2 The intervention

`weight_correlation.py`, rate-calibrated (39.7–40.0 Hz):

| scheme | cos | `c` (N=512) | Λ (N=512) | SNR out (N=128) | SNR out (N=512) |
|---|---:|---:|---:|---:|---:|
| gaussian | 0.901 | 0.444 | 0.94 | 3.44 | 3.46 |
| uniform / scaled (shipped) | 0.750 | 0.362 | 2.01 | 3.92 | 4.07 |
| **sparse** (ρ = 0.1) | **0.075** | **0.203** | **3.88** | **4.68** | **5.31** |

Measured cosine 0.7525 against 0.7519 predicted at N = 128. Ranking monotone in cosine at both
widths. Sparse `[512,512]` reaches **5.31 dB against the shipped `[32,16,8]` ceiling of
1.74 dB** (§8.6.3) — **+3.57 dB** on real patient ECG. M4 (`docs/m4_findings.md`) shows this is
decorrelation, and that K ≈ 16 inputs per neuron does better still (5.43 dB).

#### 8.6.3 The encoder-rate control

`calibrate_gain` holds each LIF layer at 40 Hz but cannot reach the encoder, whose rate followed
each record's amplitude distribution. `calibrate_encoder_rate` (new in
`network/calibration.py`) sets the encoder's ceiling per drive so its mean rate is held too.
`run_m3.py --encoder-rate 40`, shipped `[32,16,8]`, paired with an uncontrolled run on the same
seeds:

| | encoder rate | encoder Λ across records | corr(rate, Λ_enc) | ceiling | encoder binds |
|---|---:|---:|---:|---:|---:|
| fixed 100 Hz ceiling | 15.6–65.2 Hz | 3.85 ± 1.14 dB | +0.39 | 1.74 ± 0.59 dB | 16 / 16 |
| encoder held at 40 Hz | 40.0–40.2 Hz | 4.05 ± 1.09 dB | — | 1.88 ± 0.60 dB | 16 / 16 |

- **The confound was real in rate and negligible in consequence.** Removing a 4× spread in encoder
  rate barely narrows the cross-record spread of encoder Λ (sd 1.14 → 1.09 dB). That spread is a
  property of the signals — their morphology and amplitude distribution — not of the operating
  point.
- **The encoder binds on every record either way**, so §8.1's "encoder binds on all 16" stands
  as a controlled result.
- **The ceiling rises slightly under control**, median +0.18 dB (Wilcoxon p = 0.018): low-rate
  records gain (record 100: 0.66 → 0.92 dB) and high-rate records lose (108: 1.76 → 1.51 dB).
- **The law is indifferent.** RMS residual at input SNR ≥ 10 dB is 0.221 dB controlled against
  0.235 dB uncontrolled (144 points each); mean residual +0.12 and +0.13 dB.

The uncontrolled run also re-establishes §8.1 at 10 seeds: ceiling 1.74 ± 0.59 dB (2 seeds:
1.71 ± 0.76), RMS residual 0.235 dB at ≥ 10 dB input.
