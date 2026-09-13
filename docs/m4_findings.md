# M4 — The decorrelation intervention: it is decorrelation, and fan-in sets the optimum

**Status:** the open question in `docs/paper_roadmap.md` R4/M4 — whether sparse initialisation
helps by decorrelating weight columns or merely by changing fan-in — is resolved, on synthetic and
real ECG.
**Source data:** `results/m4_decorrelation/density_sweep_synthetic.csv` (10 seeds) and
`results/m4_decorrelation/density_sweep_mitbih.csv` (16 records × 10 seeds).
**Reproduce:** `uv run python scripts/m4_decorrelation/density_sweep.py [--dataset mit-bih]`.

---

## 1. The question

M3 traced the ceiling to the cosine similarity of weight columns. Every non-negative uniform
initialisation gives `cos → 3/4` independently of scale, and `init_weights(scheme="sparse")`
lowers it, lifting `[512, 512]` from 3.87 dB to 5.00 dB on synthetic ECG.

Sparsifying changes two things at once. It decorrelates the columns — at density `ρ` the cosine
of a uniform draw falls to `3ρ/4` — and it shrinks each neuron's fan-in to `K = ρN`. The two
predict different behaviour:

| mechanism | at fixed width, as K falls |
|---|---|
| decorrelation | Λ rises monotonically, because `cos = 3K/4N` falls with it |
| fan-in | each neuron averages fewer input trains, its drive gets noisier, and Λ eventually falls |

The library's Bernoulli mask cannot separate them: fan-in is binomial, and at `ρ = 0.1, N = 32`
a neuron sees about three inputs. `density_sweep.py` builds the mask with an **exact** fan-in per
neuron and sweeps K at each fixed width, so the two variables move independently. Everything else
is M3's protocol: clean input, per-layer rate calibration to 40 Hz, error correlation by
residualising each neuron's trace against the clean signal.

## 2. Synthetic ECG

Output SNR (affine-corrected, dB) at the second layer, mean over 10 seeds; seed-to-seed sd ≤ 0.19
dB in every cell (median 0.10). Firing rate held at 39.8–40.8 Hz throughout.

| K | `[32,32]` | `[128,128]` | `[512,512]` |
|---:|---:|---:|---:|
| 4 | 2.97 | 4.32 | 4.87 |
| 8 | **3.14** | **4.52** | 5.05 |
| 16 | 3.04 | 4.47 | **5.05** |
| 32 | 2.71 (dense) | 4.33 | 4.99 |
| 64 | | 4.10 | 4.88 |
| 128 | | 3.64 (dense) | 4.71 |
| 256 | | | 4.41 |
| 512 | | | 3.83 (dense) |

Measured column cosine matched `3K/4N` to within 0.005 in every cell.

**Three results.**

1. **There is an interior optimum at every width.** Going from dense down to K ≈ 8–16 gains
   +0.43 dB at N = 32, +0.88 dB at N = 128 and +1.22 dB at N = 512. Below that, the gain reverses.
   Both mechanisms are real, and they act in opposite directions.

2. **The optimum sits at a near-constant fan-in, not a constant density.** K* = 8, 8, 16 across a
   16× range of width, i.e. `ρ*` = 0.25, 0.06, 0.03. The practical rule is therefore *"give each
   neuron on the order of ten inputs"*, not *"use density 0.1"*. The library default
   (`density=0.1`, K ≈ 13 at N = 128 and K ≈ 51 at N = 512) is near the optimum at N = 128 and slightly too dense at N = 512, by about 0.1 dB.

3. **Decorrelation acts through correlation; fan-in acts through per-neuron noise.** Regressing
   the measured error correlation on both variables across all 14 conditions gives

       c = 0.264 + 0.044·log₂K − 0.036·log₂N        (R² = 0.853)

   — equal and opposite coefficients, which is what `cos = 3K/4N` requires. A model on cosine
   alone explains nearly as much (`c = 0.332 + 0.041·log₂cos`, R² = 0.839). Fan-in adds almost
   nothing to the *correlation*; its cost appears in each neuron's own variance, which is exactly
   what produces the turnover in (1). M3's mechanism is confirmed with the confound removed.

**The M3 caveat is settled.** The roadmap worried that sparse numbers at N = 32 were fan-in
limited. They were, slightly: K = 4 at N = 32 scores 0.17 dB below the optimum. The N ≥ 128
numbers M3 reported are unaffected.

## 3. Real MIT-BIH

16 records × 10 seeds, same grid. Output SNR (dB), mean across records of per-record means; the
± is the across-record spread in absolute level, which is large because records differ in how
reconstructable they are. The paired differences below are what carry the test.

| K | `[32,32]` | `[128,128]` | `[512,512]` |
|---:|---:|---:|---:|
| 4 | 3.05 | 4.51 | 5.23 |
| 8 | **3.21** | **4.70** | 5.40 |
| 16 | 3.19 | 4.70 | **5.43** |
| 32 | 2.93 (dense) | 4.58 | 5.36 |
| 64 | | 4.38 | 5.22 |
| 128 | | 3.93 (dense) | 5.02 |
| 256 | | | 4.70 |
| 512 | | | 4.08 (dense) |

Rate 39.7–40.4 Hz throughout; measured cosine identical to synthetic (pure weight algebra).

**Optimum fan-in minus dense, paired on record, Wilcoxon, Holm across the three widths:**

| width | optimum K | median gain | range across records | records improved | p (Holm) | per-record best K |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 8 | +0.30 dB | [+0.12, +0.46] | 16 / 16 | 9 × 10⁻⁵ | 8 (9 records), 16 (7) |
| 128 | 8 | +0.73 dB | [+0.42, +1.38] | 16 / 16 | 9 × 10⁻⁵ | 8 (7), 16 (9) |
| 512 | 16 | +1.20 dB | [+0.55, +2.86] | 16 / 16 | 9 × 10⁻⁵ | 8 (4), 16 (9), 32 (3) |

**Everything replicates.** The interior optimum sits at the same K at every width; every record
improves at every width; every record's own optimum lies in K ∈ {8, 16, 32}. The gain grows with
width, as the model requires.

The correlation regression on real data:

    c = 0.235 + 0.047·log₂K − 0.033·log₂N        (R² = 0.901)
    c = 0.346 + 0.043·log₂cos                     (R² = 0.855)

Cosine alone accounts for most of what `K` and `N` explain together. The coefficients are less
evenly matched than on synthetic (0.047 against −0.033), so on real signals there is a small
residual dependence of correlation on width beyond what cosine captures. It does not change the
reading — the correlation is set by column overlap — but "equal and opposite" is a synthetic-data
statement and the manuscript should not make it for MIT-BIH.

`[512,512]` at K = 16 reaches **5.43 dB against the shipped `[32,16,8]` ceiling of 1.74 dB**
(`docs/m3_findings.md` §8.6.3): **+3.69 dB** from width plus decorrelation at the right fan-in,
of which +1.35 dB (mean; median +1.20) is the decorrelation at fixed width.

## 4. What this changes

- The intervention is now stated as a design rule with a mechanism: **decorrelate the weight
  columns, but keep fan-in near ten.** The gain grows with width because the dense cosine is
  fixed at 3/4 while the optimal cosine falls as 1/N.
- Related work: Kanitscheider et al. (2015, Eq. 4) give the correlation between two neurons as
  proportional to the normalised overlap of their input filters — the same quantity as our column
  cosine (`docs/related_work.md` A2). What M4 adds is the fan-in trade-off that bounds how far
  that overlap can usefully be reduced in a spiking layer.
- **Not claimed:** that decorrelation raises information above the input's. The cascade law's
  `1/SNR_in` term is untouched by any connectivity; M4 reduces the intrinsic `1/Λ` terms only.
