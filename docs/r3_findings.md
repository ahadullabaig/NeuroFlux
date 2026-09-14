# R3 — Ceiling characterisation

**Status:** complete — C1, C2 and C3, with four post-hoc checks labelled as such (§1.5, §1.6, §2.6, §3.3).

**Code:** `scripts/r3_characterisation/` — one pipeline (`r3_common.py`), one grid per
question (`c1_flatness.py`, `c2_invariances.py`, `c3_operating_region.py`), the
pre-registered statistics (`analyse_r3.py`) and the IEEE figures (`figures_r3.py`).
Every analysis rule is written in the docstring of the grid script that produced the data,
and was written before that grid ran. Anything decided after seeing data is labelled
**post hoc** where it appears.

**Reproduce:**

```bash
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c1_flatness.py
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c1_operating_point.py
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c2_invariances.py
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c2_training_mechanism.py
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c3_operating_region.py --part rate
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c3_operating_region.py --part plane
OMP_NUM_THREADS=1 uv run python scripts/r3_characterisation/c3_operating_region.py --part rate-signals
uv run python scripts/r3_characterisation/upstream_checks.py
uv run python scripts/r3_characterisation/analyse_r3.py      # every table in analysis/
uv run python scripts/r3_characterisation/figures_r3.py      # Figs. 1-3, Table I
```

About 2.5 hours on 16 cores for all grids; the analysis and figures take minutes.

Grids resume from whatever their CSV already holds. The raw grids are committed gzipped
(`results/r3_characterisation/*.csv.gz`, 7.7 MB; the plain CSVs are 66 MB and gitignored). The
analysis reads the gzipped copies directly and reproduces every table in `analysis/` byte for
byte; to resume or extend a grid, `gunzip -k` its file first.

---

## 0. What changed in the measurement before any R3 number was taken

Four things, each found while building the pipeline, each fixed before the grids ran.

1. **The input-SNR axis now spans what it claims.** `SignalGenerator.generate` adds flicker
   at a fixed 0.5 amplitude and then scales Gaussian noise to the power of the *flickered*
   signal, so the flicker alone capped the input at about −6 dB against the clean signal
   whatever the nominal setting. `synthetic.add_interference` scales the whole interference
   against the clean signal and rescales the realised draw to its exact target variance, so
   `calculate_snr(clean, noisy)` *equals* the nominal SNR. **Flicker is kept, as an
   interference type compared at matched SNR** (§1.4), rather than dropped.
2. **`calibrate_gain` no longer discards training.** `measure_layer_rates` called
   `network.reset_state()`, which restores the *initial* weights, so calibrating a trained
   network silently reverted it. No published number is affected — every earlier script
   calibrated before training — but C2's design (train, then recalibrate so a learning knob
   changes weights and nothing else) would have measured an untrained network at every
   epoch count. It now resets transient state only; a regression test fails on the old
   code.
3. **`calibrate_gain` accepts one target rate per layer**, so an output layer can be held
   sparse while the layer feeding it stays at 40 Hz (C3's rate axis).
4. **Degeneracy has a verdict, and saturation is judged from rates.** A population pinned at
   the refractory limit is *not* constant — its readout ripples at the refractory period —
   so an output-spread guard never flags it (shipped `[256,16,8]` probe: output std 0.17 of
   the signal's, affine SNR 0.0006 dB, `is_degenerate` False). Every R3 row carries
   `verdict` ∈ {`ok`, `silent`, `saturated`}: saturated when at least half the output
   neurons fire at ≥ 90% of the rate limit (125 Hz at dt = 4 ms, measured), silent when the
   output is constant.

Streams are also separated: noise, encoder and calibration draws come from one
`SeedSequence` per seed. Earlier scripts seeded the noise generator and the encoder
generator with the same integer, which makes them the same bit stream.

---

## 1. C1 — Flatness

**Grid** (`results/r3_characterisation/c1_flatness.csv`, 12,060 runs, 0 failures): synthetic
ECG, the 16 Phase 3 MIT-BIH records, synthetic EEG and a 2 Hz sine; input SNR −20 to +40 dB
in 5 dB steps plus a noiseless input; 10 seeds (a different artifact-free window per seed on
MIT-BIH); three arms:

| arm | init | rates | training |
|---|---|---|---|
| rate-controlled, untrained | scaled | encoder and both layers held at 40 Hz | none — M3's operating point |
| shipped, untrained | uniform | free; one global `g_syn = 30`, encoder ceiling 100 Hz | none |
| shipped, trained | uniform | free | 10 STDP epochs |

The unit is a record on MIT-BIH (n = 16) and a seed on the synthetic signals (n = 10).

### 1.1 The ceiling is flat — acceptance met on all four signals

Least-squares slope of affine-corrected output SNR on input SNR, per unit; median across
units with a 95% BCa interval. The unprocessed input's own slope is the reference a lossless
pipeline would follow.

| signal | arm | slope, −20 to 40 dB | slope, 10 to 40 dB |
|---|---|---|---|
| synthetic ECG | rate-controlled | +0.035 [+0.032, +0.036] | +0.013 [+0.006, +0.017] |
| | shipped, untrained | +0.024 [+0.023, +0.026] | +0.010 [+0.008, +0.014] |
| | shipped, trained | +0.033 [+0.027, +0.036] | −0.002 [−0.013, +0.009] |
| MIT-BIH | rate-controlled | +0.040 [+0.027, +0.047] | +0.012 [+0.005, +0.014] |
| | shipped, untrained | +0.034 [+0.021, +0.038] | +0.011 [+0.007, +0.014] |
| | shipped, trained | +0.022 [+0.016, +0.035] | +0.002 [+0.001, +0.004] |
| EEG | rate-controlled | +0.048 [+0.043, +0.057] | +0.026 [+0.016, +0.037] |
| | shipped, untrained | +0.047 [+0.044, +0.053] | +0.042 [+0.032, +0.054] |
| | shipped, trained | +0.016 [+0.011, +0.019] | +0.005 [−0.001, +0.015] |
| sine | rate-controlled | +0.154 [+0.151, +0.157] | +0.092 [+0.081, +0.104] |
| | shipped, untrained | +0.111 [+0.109, +0.117] | +0.069 [+0.065, +0.079] |
| | shipped, trained | −0.002 [−0.005, −0.001] | −0.000 [−0.004, +0.001] |
| *unprocessed input* | | *+0.711* | *+0.989* |

**Every interval excludes 1.0**, which is the roadmap's acceptance criterion. Above 10 dB,
where the input itself improves one-for-one, the network's output improves by at most
0.09 dB per dB and on ECG by about 0.01.

Finding 1 (three seeds, synthetic ECG, the capped axis) reported an affine slope of +0.108.
On an axis that spans its range the slope is a third of that.

### 1.2 The ceiling depends on the signal

Output SNR with a noiseless input — the ceiling itself — median [IQR] across units:

| signal | rate-controlled | shipped, untrained | shipped, trained |
|---|---|---|---|
| synthetic ECG | 1.76 [1.65, 1.79] | 1.19 [1.17, 1.25] | 2.02 [1.64, 2.12] |
| MIT-BIH | 1.92 [1.38, 2.27] | 1.65 [1.08, 1.83] | 1.25 [0.86, 1.70] |
| EEG | 2.29 [2.16, 2.40] | 2.31 [2.13, 2.61] | 0.88 [0.77, 1.11] |
| sine | **7.48** [7.39, 7.58] | 5.57 [5.54, 5.75] | 0.07 [0.05, 0.08] † |
| output rate (median) | 40 Hz | 58–75 Hz | 0.9–8.9 Hz; 0.0 on sine |

† 30% of the sine's noiseless trained runs are silent; the value is over the rest.

- **The ceiling is a property of the architecture *and the signal*, not a universal number.**
  A 2 Hz sine reaches 7.5 dB through the same network that holds ECG under 2 dB. That is
  what the cascade law predicts — each stage's intrinsic SNR is a signal variance over a
  noise variance, and a smooth, band-limited signal puts more of its variance where a
  20 ms readout can follow it — but it means "the ceiling" must always be quoted with its
  signal.
- **Finding 5's signal table describes trained networks, not the architecture.** It reported
  EEG 0.08 ± 0.06 and sine 0.08 ± 0.07 dB, measured after 10 STDP epochs. C1 reproduces the
  sine collapse under training (0.07 dB, 30% of runs silent) but the untrained network reaches
  7.48 dB at the controlled operating point and 5.57 dB shipped, so it is the learning rule
  that silences a periodic signal, not the pipeline's capacity. EEG is 0.88 dB trained and
  2.29 dB untrained here; Finding 5's lower 0.08 dB came from a different protocol (5 s,
  flicker on, 3 seeds) and is not reproduced.
- **The MIT-BIH ceiling replicates M3.** 1.92 dB here against 1.88 ± 0.60 dB in
  `docs/m3_findings.md` §8.6.3, from independent code, 10 s rather than 8 s windows and
  separated random streams.
- **The trained shipped network runs sparse.** Ten STDP epochs take the output from
  58–65 Hz to about 1 Hz on ECG. On synthetic ECG that raises the ceiling (2.02 against
  1.19 dB untrained); on real ECG it lowers it (1.25 against 1.65). On the sine it silences
  the output. Output rate is therefore not a nuisance variable — C3 measures it directly.
- **Degeneracy.** 46 of 12,060 runs are not `ok`, all silent, all trained shipped (26 sine,
  20 MIT-BIH). No run saturated.

### 1.3 The cascade law — and a correction to how it was applied

The pre-registered test takes each unit's ceiling from its noiseless run and predicts the
output at every other input as `cascade_snr_db(input, [ceiling])`, on affine-corrected
values, as M3 did. Residual = measured − predicted:

| signal | arm | form | input ≥ 10 dB: mean, RMS | input < 10 dB: mean, RMS |
|---|---|---|---|---|
| MIT-BIH | rate-controlled | affine values (pre-registered) | +0.04, 0.09 | **+1.71, 1.82** |
| | | S/N (post hoc) | −0.07, 0.14 | −0.30, 0.39 |
| | shipped, untrained | affine values | +0.01, 0.09 | +1.76, 1.88 |
| | | S/N | −0.09, 0.16 | −0.32, 0.41 |
| | shipped, trained | affine values | +0.10, 0.19 | +2.07, 2.13 |
| | | S/N | −0.00, 0.09 | −0.08, 0.15 |
| synthetic ECG | rate-controlled | affine values | −0.01, 0.11 | +1.73, 1.82 |
| | | S/N | −0.12, 0.18 | −0.31, 0.38 |

**Below 10 dB the pre-registered form fails by about 2 dB, and the reason is the metric, not
the network.** Affine-corrected SNR fits the best scale and offset, leaving a residual of
`var(x)(1 − ρ²)`, so it equals `1/(1 − ρ²)` — and for a signal plus independent noise that is
exactly `1 + S/N`. It cannot fall below 0 dB. The cascade law is derived for `S/N` itself
(`metrics/cascade.py`), so fed affine values it predicts outputs the metric cannot produce:
a 1.8 dB-ceiling pipeline at −20 dB input is predicted at about −2 dB.

**Post hoc, and parameter-free:** convert to `S/N = SNR_aff − 1` before applying the law and
back afterwards (`affine_to_excess_db` / `excess_to_affine_db`, added to
`metrics/cascade.py` with a test that checks the identity on data). Below 10 dB this cuts the
RMS residual from 1.82 to 0.39 dB on MIT-BIH and from 1.82 to 0.38 dB on synthetic ECG. At
10 dB and above the two forms agree to within 0.1 dB, and there the pre-registered form is
marginally the better fit (RMS 0.09 against 0.14 dB on MIT-BIH, rate-controlled).

**What this withdraws from M3.** `docs/m3_findings.md` §3 and §8.1 read the same low-SNR
residual — +1.58 dB at 3 dB input on MIT-BIH — as the network doing *better* than the law,
attributed it to "the encoder's clip to `[0, 1]`" acting as a limiter, and called it "the
Phase 3 regime advantage, derived". Neither half holds:

1. In `S/N` terms the residual is not positive. The network does 0.3 dB *worse* than the law
   below 10 dB, not 1.6 dB better.
2. There is no limiter. Every pipeline sets the encoder's `v_min`/`v_max` to the drive's own
   extremes, so normalised amplitudes lie in `[0, 1]` by construction and the amplitude clip
   never engages; at a 100 Hz ceiling and dt = 4 ms the spike probability peaks at 0.4, so
   the probability clip never engages either.

Phase 3's regime advantage itself — raw ΔSNR improving at input ≤ 0 dB — is a measurement
on raw SNR and is unaffected. What is withdrawn is M3's derivation of it. The QRS-F1
advantage at ≤ 0 dB (`docs/r2_findings.md` §5) is annotation-referenced and also unaffected.

**What remains.** The `S/N`-form residual is structured, not noise: −0.3 to −0.6 dB between
0 and 10 dB on ECG, larger on EEG (mean −0.51 dB below 10 dB) and the sine (−0.60 dB below
10 dB; −0.69 dB at 10 dB and above). The trained arm — whose weights adapt to the input —
fits best. A post-hoc diagnostic tests whether noise moving the operating point explains it
(§1.5).

### 1.4 Flicker is handled no worse than broadband noise

Output SNR with a 50 Hz flicker, or an equal-power flicker/noise mix, minus output SNR with
Gaussian noise at the same input SNR; paired by unit, Holm across the six input SNRs, TOST
at ±0.5 dB.

| signal | arm | flicker − Gaussian, median (dB) | verdict |
|---|---|---|---|
| MIT-BIH | rate-controlled | +0.12, +0.21, +0.21, +0.14, +0.06, +0.03 at −10…20 dB | equivalent at every SNR |
| | shipped, trained | +0.25, **+0.39**, **+0.36**, +0.09, +0.07, +0.00 | better at −5 and 0 dB (p_holm = 0.0002; not equivalent); equivalent elsewhere |
| synthetic ECG | rate-controlled | +0.11 to +0.19 | equivalent at every SNR |
| | shipped, trained | +0.44, **+0.78**, **+0.74**, +0.02, +0.06, −0.12 | better at −5 and 0 dB (p_holm = 0.012); equivalent elsewhere |

The mix is equivalent to Gaussian noise everywhere (|median| ≤ 0.26 dB).

Periodic interference at a given power costs the untrained network slightly less than white
noise at that power — significantly so, but inside the margin — and the trained network
recovers up to 0.8 dB more from it at low input SNR. The effect is far smaller than the
network's low-pass readout would suggest (a 20 ms exponential readout alone attenuates a
50 Hz tone's power by 16 dB, and the 20 ms membrane by as much again), which is consistent
with the cost of interference being paid at the encoder's range rather than in the readout;
§1.5 tests that directly.

### 1.5 Post hoc: the law is exact once the operating point is measured where the input sets it

**Designed after §1.3's residual was seen** (`scripts/r3_characterisation/c1_operating_point.py`,
5,960 runs, 0 failures). The law takes each stage's intrinsic SNR from one noiseless run and
assumes it holds at every input. In this pipeline it cannot: the encoder's range is the noisy
drive's own minimum and maximum, and the encoder-rate and layer-gain calibrations are fitted to
that drive, so more noise means a wider range and a signal squeezed into less of it. The test
runs the network on the **noiseless** signal with its whole operating point — encoder range,
encoder rate, layer gains — taken from the same seed's drive at each input SNR, and predicts
the noisy run from *that* ceiling instead of the ordinary one.

RMS residual (dB), `S/N` form, ordinary noiseless intrinsics → operating-point-matched:

| signal | arm | input ≥ 10 dB | input < 10 dB |
|---|---|---|---|
| MIT-BIH | rate-controlled | 0.14 → **0.02** | 0.39 → **0.04** |
| | shipped, untrained | 0.16 → **0.02** | 0.41 → **0.04** |
| | rate-controlled, flicker | 0.17 → 0.02 | 0.29 → 0.09 |
| synthetic ECG | rate-controlled | 0.18 → 0.07 | 0.38 → 0.07 |
| | shipped, untrained | 0.16 → 0.04 | 0.41 → 0.04 |
| EEG | rate-controlled | 0.39 → 0.07 | 0.66 → 0.04 |
| | shipped, untrained | 0.66 → 0.05 | 0.85 → 0.03 |
| sine | rate-controlled | 0.95 → 0.18 | 0.90 → 0.23 |
| | shipped, untrained | 0.89 → 0.13 | 0.89 → 0.21 |

Maximum absolute residual on MIT-BIH, rate-controlled, over 208 unit-level points from −20 to
+40 dB: **0.11 dB**.

**The inverse-SNR cascade law is exact** — to a few hundredths of a dB on real ECG and on EEG,
across a 60 dB input range, with nothing fitted — once it is applied in `S/N` terms and its
intrinsic SNRs are measured at the operating point the input itself sets. The structured
residual in §1.3 was not a failure of additivity; it was the operating point moving with the
noise. The sine improves fivefold but keeps a +0.15 dB bias below 10 dB, so something beyond
the operating point contributes there; that is unexplained.

**How large the operating-point cost is.** The noiseless ceiling at the operating point an
input sets, minus the ordinary ceiling, median over units:

| input SNR (dB) | −20 | −10 | 0 | 10 | 20 | 30 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIT-BIH, rate-controlled, Gaussian | −1.90 | −1.77 | −1.11 | −0.34 | −0.05 | −0.00 | +0.00 |
| MIT-BIH, rate-controlled, 50 Hz flicker | | −1.52 | −0.79 | −0.24 | −0.02 | | |
| sine, rate-controlled, Gaussian | −7.43 | −7.21 | −5.84 | −2.93 | −0.82 | −0.19 | −0.09 |

At −20 dB the operating point alone removes essentially the whole ceiling: the encoder's range
is set by noise, and the signal occupies almost none of it.

- **This explains §1.4's flicker result.** A tone of a given power widens the drive's min/max
  range less than Gaussian noise of that power, whose tails set the extremes. At 0 dB on MIT-BIH
  flicker's operating point costs 0.79 dB against Gaussian's 1.11; the 0.32 dB difference is the
  size and sign of the +0.21 dB flicker advantage C1 measured, and the same holds at −5 and
  −10 dB. The network does not reject periodic interference; its encoder is squeezed less by it.
- **It is a design lever.** Up to 1.1 dB at 0 dB input on MIT-BIH, and 5.8 dB on the sine, is
  lost to how the encoder's range is chosen rather than to spiking noise. A range estimate that
  does not grow with the noise — a robust percentile, or a running estimate of signal scale —
  could recover part of it. Untested; a candidate for R4.
- **For the manuscript,** Section III can state the law as exact rather than as a high-SNR
  approximation, provided it is stated with both conditions: `S/N` terms, and intrinsics at the
  input's operating point.

### 1.6 What the confound check found upstream of R3

The roadmap's C1 item named the flicker confound for synthetic ECG. Checking it on the
MIT-BIH path turned up two things about results *before* R3. Both are descriptive
measurements, taken after C1, and neither is a pre-registered test.

**The Phase 3, R1 and R2 input-SNR labels are nominal.** `mit_bih.annotator.make_pair` uses
the same construction as `SignalGenerator`: flicker at 0.5 amplitude first, then Gaussian
noise scaled to the power of the flickered signal. MIT-BIH segments are normalised to unit
variance, so the flicker alone sits at +9 dB and caps the input there. Measured with
`calculate_snr(clean, noisy)` on `make_pair` output, 16 records × 3 seeds, 10 s windows:

| nominal (as labelled) | −5 dB | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---:|---:|---:|---:|---:|
| true input SNR, median | **−5.62** | **−0.93** | **+3.21** | **+6.28** | **+8.68** |
| range across 48 windows | −5.68 to −5.61 | −0.97 to −0.91 | +3.19 to +3.25 | +6.24 to +6.31 | +8.65 to +8.69 |

The spread is tiny because it is algebra, not noise: `1 / (0.125 + 1.125 · 10^(−s/10))`.
Nothing reported in those documents changes direction. R1's saturation result still stands:
`rpeak_integrity` moves 0.000 across what is really a 14.3 dB span. Phase 3's regime boundary
becomes "improves at ≤ −0.9 dB, degrades at ≥ +3.2 dB", and its top input class is +8.7 dB,
not 20. Correction banners are in `docs/r1_findings.md`, `docs/r2_findings.md` and
`docs/phase3_findings.md`.

**The "doing nothing" QRS baseline never removes the flicker.** `metrics/qrs.detect_rpeaks`
passes the signal straight to `neurokit2.ecg_peaks`, which does not clean it. On C1's own rows
(MIT-BIH, true input SNR, median F1 over records of the unprocessed input) the raw detector
reaches 0.89 at 5 dB and 1.00 at 10 dB with Gaussian noise, but only 0.11 and 0.93 with the
same power in a 50 Hz tone. Add neurokit's own standard cleaning (`ecg_clean`: 0.5 Hz
high-pass plus a powerline filter) and, on the Phase 3 pairs:

| nominal input | raw detector | after `ecg_clean` | best NeuroFlux median (R2 sweep) |
|---|---:|---:|---:|
| −5 dB | 0.024 | **0.917** | 0.515 (10 epochs) |
| 0 dB | 0.093 | **1.000** | 0.637 (10 epochs) |
| 5 dB | 0.507 | **1.000** | 0.633 (10 epochs) |
| 10 dB | 0.811 | **1.000** | 0.690 (10 epochs) |
| 20 dB | 0.910 | **1.000** | 0.708 (10 epochs) |

Median over 16 records of each record's mean over 3 seeds; R2's column is its 10-seed sweep.
Individual records reach 0.98 under the network.

So R2 §5's "the 10-epoch network beats doing nothing at ≤ 0 dB" is a comparison against a
detector that skips the step every ECG pipeline takes, and a zero-parameter filter recovers
beats better than any NeuroFlux configuration measured. **This does not bear on the ceiling
result**, which is a statement about reconstruction limits of an architecture class, not a
clinical-utility claim. It does bear on anything the manuscript says about QRS recovery, on
Phase 4 (whose baselines must include this filter), and on R5's detector-robustness check
(which should score every condition with and without `ecg_clean`).

Reproduce: `uv run python scripts/r3_characterisation/upstream_checks.py`.


---

## 2. C2 — Invariances

**Grid** (`results/r3_characterisation/c2_invariances.csv`, 6,120 runs scored at six decoder
time constants each, 0 failures): rate-controlled operating point (encoder and every layer
held at 40 Hz, recalibrated after any training); MIT-BIH (16 records) and synthetic ECG
(10 seeds); a noiseless input and a 10 dB input. Training was also run at the shipped
operating point. Every level is paired with its reference by unit; Wilcoxon, Holm within
each knob's family, BCa interval on the median change, TOST at ±0.5 dB (±0.05 for QRS F1).
Holding rates worked: output rate 39.5–41.2 Hz at every trained level. Every rate-controlled
run is `ok`; 5 of 6,120 runs are silent, all shipped MIT-BIH networks after 10 epochs (4 at
10 dB input, 1 noiseless).

**Table I** (`results/r3_characterisation/figures/table1_invariances.tex`) and **Fig. 2**
carry the full comparison. The table and Figs. 1–3 compile with pdfTeX (TeX Live 2026) in a
two-column stand-in with IEEE journal geometry (43 pc text block, 1 pc column gap, 10 pt Times),
with no overfull boxes; `IEEEtran.cls` itself is not installed here, so its caption and heading
styles are unchecked.

### 2.1 Verdicts, as pre-registered

A knob is *invariant* only if every level is equivalent to the reference; it *matters* if any
level differs significantly by at least the margin; otherwise it is inconclusive.

| knob | MIT-BIH ceiling | synthetic ECG ceiling | reading |
|---|---|---|---|
| output width N, 4–64 | **invariant** | **invariant** | the claim survives, now with equivalence tests |
| STDP τ, 5–80 ms (10 epochs) | **invariant** | inconclusive (80 ms: +0.31, significant, CI to +0.43) | survives on real ECG |
| decoder τ, 5–160 ms | matters | matters | optimum at 20–40 ms; nothing above it |
| depth, 2–4 layers | matters | matters | as the cascade law predicts |
| training, 0–200 epochs, rates held | **matters — lowers it** | matters — lowers it | new |

The same verdicts hold at 10 dB input, except that STDP τ becomes invariant on synthetic ECG
as well.

### 2.2 Output width: invariant, and not for lack of power

MIT-BIH ceiling by output width, median over 16 records, against what independent output
neurons would give. The prediction takes each record's own N = 8 run, infers the output
layer's intrinsic `S/N`, scales it by N/8, and runs the cascade law — no fitted parameters:

| N | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| measured (dB) | 1.84 | 1.92 | 1.96 | 1.98 | 1.99 |
| independent-neuron prediction (dB) | 1.44 | 1.92 | 2.30 | 2.56 | 2.71 |

Sixteen times the output population buys **+0.08 dB** (paired median, 64 against 8,
[+0.06, +0.09]; every level equivalent), where independent neurons would buy +0.79. Even the
independent-neuron prediction cannot pass 2.88 dB, the SNR arriving from the hidden layer,
which is why it is far below the +12 dB Finding 2 quoted: widening one stage is capped by the
stages before it. The fitted shared-error correlation is `c` = 0.48 [IQR 0.45, 0.51] on
MIT-BIH, 0.48 on synthetic ECG (descriptive; M3 measured 0.35–0.44 on layer 1 of `[N, N]`).

### 2.3 The decoder window has an optimum and cannot lift the ceiling

MIT-BIH ceiling: 0.81, 1.11, **1.92**, **1.98**, 1.34 and 0.76 dB at τ = 5, 10, 20, 40, 80
and 160 ms. 40 ms is equivalent to 20 (+0.21 [−0.06, +0.26]); 5, 10 and 160 ms are 0.75–1.06
dB worse. Finding 2 reported the change between the two ends of the range (−0.23 dB); the
cost of a mistuned window against the optimum is up to 1.06 dB. Its reading stands in the form
that matters: **no decoder window lifts the ceiling past
the margin** — the best level is +0.21 dB. A too-short window passes spike noise, a too-long
one smooths the QRS away. QRS F1 is ≈ 0 at every τ, because the output fires at 40 Hz (§3).

### 2.4 Training lowers the ceiling at a held rate

| epochs | 0 | 10 | 50 | 200 |
|---|---:|---:|---:|---:|
| MIT-BIH ceiling, rates held (dB) | 1.92 | 1.34 | 1.13 | 1.06 |
| change, [95% CI] | — | −0.49 [−0.60, −0.39] | −0.60 [−0.92, −0.51] | −0.64 [−0.75, −0.56] |
| MIT-BIH ceiling, shipped (dB) | 1.65 | 1.25 | 1.16 | 0.93 |
| shipped output rate (Hz) | 64.8 | 1.1 | 4.3 | 5.0 |
| shipped QRS F1, 10 dB input | 0.05 | **0.68** | 0.52 | 0.51 |

Synthetic ECG, rates held: −0.52, −0.68, −0.72 dB. At 10 dB input, MIT-BIH: −0.48, −0.68,
−0.72.

- **Learning cannot lift the ceiling; at a fixed operating point it lowers it**, by about
  0.5 dB in the first 10 epochs and a further 0.15 dB by 200. That is stronger than Finding 2's
  "STDP moves it by fractions of a dB" and consistent with the architecture claim — but it is
  not invariance, and the manuscript cannot call it that.
- **As deployed, training mostly changes the operating point.** With rates free, 10 epochs
  collapse the output from 65 Hz to 1 Hz. That is what raises QRS F1 from 0.05 to 0.68 while
  the ceiling falls (MIT-BIH) or rises (synthetic ECG, +0.79 dB). Held at 40 Hz, the same
  training leaves F1 at 0.02. The QRS benefit of STDP is the sparse output, not the learned
  weights — which is what C3's rate axis tests directly.
- A post-hoc check of M3's mechanism for the loss — that Hebbian learning on a tiled input
  makes weight columns more alike — is in §2.6.

### 2.5 Depth and STDP time constant

**Depth** follows the cascade law's one unconditional prediction, that every stage adds loss:
2, 3 and 4 layers give 2.75, 1.92 and 1.57 dB on MIT-BIH (+0.91 and −0.36 dB against 3; the
4-layer loss is significant but inside the margin). Removing a layer is worth more than any
other knob tested.

**STDP τ** (10 epochs, rates held) is equivalent to 20 ms at every level on MIT-BIH, with a
shallow U: 1.62, 1.42, 1.34, 1.35 and 1.52 dB at 5–80 ms. Every trained level stays below
the untrained 1.92 dB.

### 2.6 Post hoc: which stage training damages, and M3's mechanism only half explains it

**Designed after §2.4 was seen** (`scripts/r3_characterisation/c2_training_mechanism.py`,
680 runs that reproduce C2's ceilings exactly). The predictions, written before it ran: STDP on
a tiled input pushes each layer's weight columns together (mean column cosine rises), and
records whose cosine rises more lose more ceiling.

| rates held at 40 Hz, MIT-BIH | 0 epochs | 10 | 50 | 200 |
|---|---:|---:|---:|---:|
| ceiling (dB) | 1.92 | 1.34 | 1.13 | 1.06 |
| layer-1 column cosine | 0.76 | **0.45** | 0.37 | 0.38 |
| layer-2 (output) column cosine | 0.76 | 0.78 | 0.80 | 0.80 |
| layer-1 mean weight | 0.15 | 0.04 | 0.04 | 0.04 |
| intrinsic `S/N`, encoder (dB) | +1.79 | +1.79 | +1.79 | +1.79 |
| intrinsic `S/N`, layer 1 (dB) | +3.69 | +0.96 | −0.52 | −0.84 |
| intrinsic `S/N`, layer 2 (dB) | +0.98 | −1.56 | −1.99 | −1.88 |

Synthetic ECG: layer-1 cosine 0.76 → 0.61, layer-2 cosine 0.75 → 0.93; layer-1 intrinsic `S/N`
+4.19 → −2.95 dB, layer 2 −0.08 → −0.71 dB.

Intrinsic `S/N` per stage is computed from C2's own recorded stage SNRs, converted to `S/N`
and inverted through the cascade law; the encoder, which STDP does not touch, is unchanged,
as it must be.

- **Prediction 1 is refuted for layer 1 and holds for the output layer.** Training
  *decorrelates* layer 1's weight columns (cosine 0.76 → 0.37) while shrinking its weights
  fourfold; it pulls the output layer's columns together, modestly on MIT-BIH and strongly on
  synthetic ECG.
- **Prediction 2 holds for the output layer only.** Across records and epoch levels, the change
  in output-layer cosine correlates with the change in ceiling at Spearman ρ = −0.33
  (p = 0.024, n = 48) on MIT-BIH and −0.67 (p = 5 × 10⁻⁵, n = 30) on synthetic ECG. Layer 1:
  ρ = +0.26 (p = 0.07) and −0.14 (p = 0.47).
- **The largest loss is in the layer whose columns decorrelated.** Layer 1 loses 4.5 dB of
  intrinsic `S/N` on MIT-BIH, the output layer 2.9 dB. M3's column-cosine account therefore
  explains part of the loss — the output layer's convergence — but not the larger part. Its
  3/4 derivation assumes independently drawn weights; after training they are not, and cosine
  stops predicting a stage's noise.
- **Untested hypothesis, for R4:** an effective fan-in collapse. With layer-1 weights shrunk
  fourfold and the gain recalibrated upward to hold 40 Hz, each neuron is driven by the few
  synapses that survived, and M4 found that too few effective inputs per neuron raises its
  noise. The participation ratio of each weight column would test this directly; it was not
  recorded.

### 2.7 What C2 does to the acceptance criterion

**Acceptance (roadmap R3):** invariance to N is supported by equivalence tests — met.
Invariance to the learning rule is met for its time constant and *refuted* for the amount of
learning, in the direction that strengthens the ceiling claim rather than weakening it. The
framing consequences of C1–C3 together are in §4.

---

## 3. C3 — The operating region

### 3.1 The gain plane: the usable band narrows, and training closes it

**Grid** (`results/r3_characterisation/c3_plane.csv`, 48,000 runs, 0 failures): the shipped
operating point — uniform init, one global `g_syn` for every layer, encoder ceiling 100 Hz —
over `N_in` ∈ {8, …, 256} × 25 gains from 1 to 300, untrained and after 10 STDP epochs, on
MIT-BIH at 10 dB input, 16 records × 10 seeds per cell. **Fig. 3 (a, b).**

**Untrained — pre-registered boundaries** (silent fraction first ≤ 0.5; saturated fraction
first ≥ 0.5; interpolated in log gain):

| N_in | silent below g | saturated above g | usable band | best SNR (dB) @ g, output rate | best QRS F1 @ g, output rate | at g = 30: `ok` / SNR / F1 |
|---:|---:|---:|---:|---|---|---|
| 8 | 11.5 | > 300 | > 4.7 oct | 1.22 @ 22, 12 Hz | 0.62 @ 17, 3.3 Hz | 1.00 / 1.03 / 0.26 |
| 16 | 8.9 | 168 | 4.2 oct | 1.56 @ 17, 19 Hz | 0.63 @ 11, 0.9 Hz | 1.00 / 1.20 / 0.06 |
| 32 | 6.9 | 84 | 3.6 oct | 2.02 @ 11, 10 Hz | 0.66 @ 8.5, 1.2 Hz | 1.00 / 1.16 / 0.03 |
| 64 | 5.3 | 43 | 3.0 oct | 2.20 @ 8.5, 13 Hz | 0.78 @ 6.7, 1.6 Hz | 0.86 / 0.86 / 0.30 |
| 128 | 4.4 | 27 | 2.6 oct | **2.65** @ 6.7, 13 Hz | 0.71 @ 5.3, 1.5 Hz | **0.46** / 0.29 / 0.54 |
| 256 | 3.4 | 23 | 2.7 oct | 2.32 @ 5.3, 12 Hz | 0.79 @ 4.2, 1.1 Hz | **0.02** / 0.01 / 0.29 |

*(The N_in = 8 saturated boundary is not reached: 12% of runs saturate at g = 300.)*

- **Finding 3, mapped.** Both boundaries fall as the input widens — silence from g = 11.5 to
  3.4, saturation from above 300 to 23 — and the band between them narrows from more than
  4.7 octaves to 2.6. The shipped gain of 30 is inside it up to N_in = 32, at its edge at 64,
  and outside it from 128, where half the runs saturate; at 256 almost all do.
- **The best gain is never the shipped one** (the nearest is 22, at N_in = 8), and it lands the
  output at a specific rate. At every width, the SNR-best gain puts the output at 10–19 Hz and
  the F1-best gain at 0.9–3.3 Hz — the same two optima the rate axis found (§3.2) by a
  completely different manipulation. At the gain nearest 30 the output fires at 22, 44, 68, 95,
  115 and 122 Hz for N_in = 8 to 256: close to the SNR optimum only at the narrowest input.
- **At its best gain, a wider tiled input does reach a higher ceiling**: 1.22 dB at N_in = 8 to
  2.65 dB at 128. M1a's width slope (+0.1 dB per doubling) was measured with rates held at
  40 Hz; this one compounds width with the output rate each width's best gain selects, so the
  two are not in conflict, but the width effect at a *matched sparse* rate is unmeasured
  (a candidate for R4).

**Trained — the boundary rule does not apply.** The pre-registered rule assumes silence only
falls as gain rises. After training it returns at high gain: STDP depresses the weights of the
layers driven hardest until the output stops firing, while layer 1 still does (121 Hz at
g = 300, N_in = 256, with the output silent). Descriptively, the gains at which at least half
the runs are `ok`:

| N_in | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| usable gains | 13.7–300 | 10.8–237 | 10.8–91 | 10.8–22 (never above 65% `ok`) | **none** | **none** |
| best SNR (dB) | 0.82 | 1.02 | 1.22 | 0.86 | — | — |
| best QRS F1 | 0.63 | 0.65 | 0.69 | 0.36 | 0.00 | 0.00 |

- **Training never beats a well-chosen gain.** At every width its best SNR is below the
  untrained best (0.82–1.22 against 1.22–2.02 dB at N_in ≤ 32), and its best F1 matches the
  untrained best (0.63–0.69 against 0.62–0.66) only up to N_in = 32.
- **At N_in ≥ 128 the shipped training regime has no working operating point at any gain.** No
  saturated region survives training either: what would have saturated is silenced instead.

**Acceptance (roadmap R3):** "the operating-region map identifies the saturated and silent
boundaries quantitatively" — met for the untrained network; for the trained network the map is
reported as usable ranges, because its silent region has two edges.

### 3.2 The output-rate axis

**Grid** (`results/r3_characterisation/c3_rate.csv`, 13,260 runs, 0 failures, every run `ok`):
rate-controlled, untrained; encoder at 40 Hz; the output layer held at 0.25–80 Hz with the
hidden layer at 40 Hz (`output`), or every layer held at the same rate (`all`); `[32,16,8]`
and, for `output`, `[32,16,64]`; MIT-BIH (16 records) and synthetic ECG (10 seeds); noiseless,
0 dB and 10 dB input. Calibration hit every target: achieved output rates 0.25–80.42 Hz on
MIT-BIH.

**MIT-BIH, median over records:**

| output rate (Hz) | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 20 | **40** | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ceiling, `output`, N = 8 (dB) | 0.50 | 0.84 | 1.57 | 2.22 | **2.40** | 2.14 | 1.82 | 1.92 | 0.97 |
| ceiling, `output`, N = 64 (dB) | 0.57 | 0.93 | 1.78 | 2.55 | **2.71** | 2.41 | 2.10 | 1.99 | 1.00 |
| ceiling, `all`, N = 8 (dB) | 0.18 | 0.28 | 0.68 | 1.63 | 1.98 | **2.05** | 1.77 | 1.92 | 1.04 |
| QRS F1 at 10 dB, `output`, N = 8 | 0.62 | 0.76 | **0.77** | 0.68 | 0.60 | 0.38 | 0.19 | 0.04 | 0.40 |
| QRS F1 at 10 dB, `output`, N = 64 | 0.80 | **0.82** | 0.76 | 0.68 | 0.60 | 0.38 | 0.17 | 0.03 | 0.40 |
| QRS F1 at 10 dB, `all`, N = 8 | 0.28 | 0.51 | 0.78 | **0.78** | 0.61 | 0.47 | 0.25 | 0.04 | 0.26 |
| QRS F1 at 0 dB, `output`, N = 8 | 0.50 | 0.63 | **0.65** | 0.61 | 0.57 | 0.35 | 0.06 | 0.00 | 0.30 |

The unprocessed input scores F1 1.00 at 10 dB and 0.22 at 0 dB with the unfiltered detector
(§1.6 for what a standard filter does).

**Synthetic ECG ceiling, `output`, N = 8:** 1.00, 1.54, 2.73, **3.78**, 3.23, 2.52, 1.84, 1.76
and 0.58 dB at 0.25–80 Hz — twice the 40 Hz value at 2 Hz. N = 64 reaches 4.15 dB at 2 Hz.

#### The sparse-output hypothesis is confirmed

Pre-registered: QRS F1 at 10 dB input is higher with the output at ≤ 2 Hz than at 40 Hz. Every
sparse rate beats 40 Hz in every arm (Holm-adjusted p = 0.0002 at every level; every interval
excludes zero): by +0.58 to +0.76 F1 when only the output layer is sparse, and by +0.21, +0.48,
+0.75 and +0.76 at 0.25, 0.5, 1 and 2 Hz when every layer is. R1 §5's four-record observation is now a
16-record, 10-seed result, and C2 §2.4 closes the loop: STDP's QRS benefit in the shipped
network (0.68 at 10 epochs) is matched or beaten by an *untrained* network held at 1 Hz (0.77),
and held at 40 Hz the same training gives 0.02. **Beat detection needs a sparse output; it does
not need learning.**

It is still a detector-free comparison: at 10 dB input no rate reaches the unprocessed signal's
1.00, and at 0 dB the best network (0.72, `all` at 1 Hz) beats the raw detector (0.22) but not
a standard ECG filter (§1.6).

#### The ceiling measured at 40 Hz does not describe the sparse regime

Ceiling at each rate against 40 Hz, paired by record, TOST ±0.5 dB, Holm across rates:

| rate | `output`, N = 8 | `output`, N = 64 |
|---|---|---|
| 5 Hz | +0.65 [+0.02, +1.27], p_holm 0.086 — inconclusive | **+0.81 [+0.17, +1.48], p_holm 0.025 — higher** |
| 10 Hz | +0.34 [+0.08, +0.76], p_holm 0.078 — inconclusive | **+0.57 [+0.25, +0.99], p_holm 0.017 — higher** |
| 20 Hz | −0.04, equivalent | +0.15, equivalent |
| 2 Hz | +0.03 [−0.44, +1.59] — inconclusive | +0.21 [−0.38, +1.71] — inconclusive |
| ≤ 1 Hz, 80 Hz | lower, significantly, except 1 Hz (inconclusive) | lower, significantly, except 1 Hz (inconclusive) |

- **Output rate is the largest operating-point variable found in R3.** The ceiling rises by up
  to 0.8 dB on MIT-BIH and 2.3 dB on synthetic ECG between 40 Hz and its best rate — no C2 knob
  except removing a layer raises it at all — and falls by 1.1 dB at 0.25 Hz and 0.7 dB at
  80 Hz.
- **The effect is bimodal across records**, which is why the intervals are wide. Per record,
  eight records (100, 101, 103, 105, 113, 119, 214, 234) gain 1.1–2.3 dB by moving the output to
  2–5 Hz; the other eight (108, 112, 200, 207, 212, 228, 231, 232) gain at most 0.3 dB, at
  10–80 Hz. The second group is mostly the 200-series arrhythmia records. This is descriptive,
  not a tested hypothesis.
- **Width matters once the output is sparse.** 64 against 8 output neurons: +0.07 dB at 40 Hz
  (C2) but +0.31 to +0.33 dB at 2–5 Hz, and F1 at 0.25 Hz rises from 0.62 to 0.80. A population
  of rare spikes has independent timing to average; a dense one does not.
- **Every ceiling here is still a ceiling.** In `S/N` terms the cascade law holds at every rate,
  and holds *better* when the output is sparse: RMS residual 0.03–0.24 dB at ≤ 1 Hz, rising to
  0.2–0.6 dB at 10–40 Hz. So the structured residual in C1 §1.3 is a dense-regime effect. Output
  quality stays flat in input quality at every rate; only the height of the plateau moves.

**Consequence for the paper.** Every ceiling number before this section — M3, M4, C1, C2 — was
measured at the 40 Hz operating point, which is not the best rate for ECG reconstruction and is
the worst rate tested for beat detection. The claim survives as "the
ceiling is flat in input quality at every operating point", but its height must be reported as
a function of output rate, and the highest measured MIT-BIH ceiling is 2.71 dB (N = 64, 5 Hz).
Whether sparse outputs help only spiky signals is tested post hoc in §3.3.

### 3.3 Post hoc: the best output rate depends on the signal

**Designed after §3.2 was seen** (`c3_operating_region.py --part rate-signals`, 360 runs,
0 failures): §3.2's output-only `[32,16,8]` arm on synthetic EEG and the 2 Hz sine, noiseless
and 10 dB input, 10 seeds. A sparse output behaves like an event detector, which suits a spiky
signal; the question is whether it helps reconstruction in general.

Ceiling (noiseless input), median over seeds:

| output rate (Hz) | 0.25 | 0.5 | 1 | 2 | 5 | 10 | 20 | **40** | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MIT-BIH (§3.2) | 0.50 | 0.84 | 1.57 | 2.22 | **2.40** | 2.14 | 1.82 | 1.92 | 0.97 |
| EEG | 0.01 | 0.01 | 0.03 | 0.13 | 0.56 | 2.28 | **2.34** | 2.29 | 1.31 |
| sine | 0.30 | 0.76 | 1.17 | 1.84 | 3.35 | 5.34 | 6.86 | **7.48** | 4.84 |

Against 40 Hz (paired by seed, Holm across rates): EEG is 1.9–2.3 dB worse at every rate up to
5 Hz and 1.1 dB worse at 80 Hz, and equivalent at 20 Hz; the sine is significantly worse at
every other rate, by 0.55 dB at 20 Hz and by 6.3 dB at 1 Hz.

**The rate optimum is a property of the signal.** Spiky ECG reconstructs best with the output
at 2–5 Hz; EEG, a sum of 10–40 Hz rhythms, is flat from 10 to 40 Hz and collapses below; the
sine peaks at 40 Hz and loses 4 dB at 5 Hz. §3.2's statement that 40 Hz is not the best rate
is **specific to ECG** — and to landmark-dominated signals generally, by this argument — and the
manuscript must scope it that way. For beat detection, which only needs the landmarks, the
sparse optimum is what one would expect on any signal; for reconstruction, the output must fire
fast enough to follow the signal's own bandwidth.

---

## 4. What R3 does to the paper

**The framing sentence.** The roadmap's reads: the ceiling is "independent of input quality,
population size, decoder integration window, and learning rule". Against C1–C3 it becomes,
as a draft for the author's review:

> Rate-coded spiking reconstruction pipelines have an output quality ceiling that is flat in
> input quality (slope ≤ 0.09 dB/dB above 10 dB input on four signals, and a cascade law that
> holds at every output rate tested). Its height is set by the signal, the depth and the output firing rate; it is not
> lifted by a wider output population at a dense output, by any decoder window, or by
> unsupervised learning, which lowers it at a held rate. An inverse-SNR cascade law predicts
> the output to within 0.04 dB RMS on real ECG from −20 to +40 dB input, with no fitted
> parameters, once it is applied to signal-to-noise power and to the operating point the input
> itself sets.

**What survives from before R3.** The ceiling (flat, replicated at 1.92 dB on MIT-BIH); width
invariance at a dense output (now with equivalence tests); the cascade law (now exact, in a
corrected form); the gain-scaling failure of Finding 3 (now mapped).

**What R3 withdraws or narrows.**

| earlier claim | status |
|---|---|
| Finding 1: slope +0.108 | superseded: ≤ 0.09 on an uncapped axis, ≈ 0.01 on ECG |
| Finding 2: decoder τ, learning "do not set the ceiling" | narrowed: neither lifts it; mistuned τ costs 1 dB; learning lowers it |
| Finding 5: EEG and sine ceilings 0.08 dB | trained-network values; untrained 2.29 and 7.48 dB |
| M3 §3: law under-predicts at low SNR because of a limiter; "Phase 3 regime advantage, derived" | withdrawn: affine metric floor; no limiter engages |
| M3 §4: column cosine caps each stage | holds for random weights; does not predict a trained layer's noise (§2.6) |
| Every ceiling number at 40 Hz (M3, M4, C1, C2) | an operating point, not the ceiling's height: ECG peaks at 2–5 Hz (§3.2), not EEG or sine (§3.3) |
| Phase 3 / R1 / R2 input SNRs | nominal; true values −5.6 to +8.7 dB (§1.6) |
| R2 §5: network recovers otherwise-undetectable QRS | only against an unfiltered detector (§1.6) |

**New open items** are recorded in `docs/paper_roadmap.md`, next actions 13–16: the mechanism
at the rate optimum, the encoder-range lever, the framing review, and filtered QRS baselines;
plus the effective-fan-in hypothesis of §2.6.
