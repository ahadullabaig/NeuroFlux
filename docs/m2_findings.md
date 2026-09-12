# M2 — locating the bottleneck

**Status:** resolved, with a correction to the first reading (see §2).
**Data:** `results/m2_bottleneck/`
**Reproduce:**
```
uv run python scripts/m2_bottleneck/run_m2.py --seeds 3 --input-snr 20.0
uv run python scripts/m2_bottleneck/run_m2.py --seeds 3 --input-snr 20.0 --tau 4.0 --layers 128 128 128
uv run python scripts/m2_bottleneck/configuration_sweep.py
```

---

## 1. Method

M1a showed the ~1 dB ceiling is not caused by the input representation. M2 asks where
inside the pipeline the information actually dies.

Every stage is scored two ways, and the *difference between the two columns* is the point:

- **Population readout** — exactly what the system does today: the `PopulationDecoder`
  arithmetic (exponential smoothing, rate conversion, clip to [0,1], affine map, mean
  across neurons) applied to that stage's spikes.
- **Optimal linear readout** — the best *any* linear decoder could do from that stage:
  least squares from per-neuron smoothed traces onto the clean signal, fit on the first
  half of the record and scored on the second, so overfitting is visible rather than hidden.

If both drop together, information is genuinely destroyed. If the linear readout stays
high while the population readout drops, the information survives and the decoder simply
cannot reach it — a fixable engineering bug rather than a limit.

All runs are per-layer rate calibrated to 40 Hz, passive (`epochs=0`), synthetic ECG,
3 seeds, 8 s, no flicker, scored with affine-corrected SNR.

## 2. Correction to the first reading

The first measurement — shipped configuration `[32,16,8]`, `tau_decode=20 ms`, 20 dB input
— gave this:

| stage | units | population | linear (test) |
|---|---:|---:|---:|
| analog input | 1 | 19.47 ± 0.03 | 19.39 ± 0.18 |
| **encoded** | 32 | **3.74 ± 0.21** (−15.73) | 3.45 ± 0.44 |
| layer1 | 16 | 2.48 ± 0.13 (−1.26) | 2.65 ± 0.15 |
| layer2 | 8 | 1.33 ± 0.13 (−1.15) | 1.30 ± 0.10 |

Read alone this says the Poisson encoder destroys 15.7 of 19.5 dB — 81% — in one step, and
that was the initial conclusion. **It is not the right conclusion.** The encoder was being
measured through a 20 ms readout at N=32, and that readout is itself most of the loss.
Characterising the encoder in isolation across `tau` and `N` shows it is not intrinsically
limited to ~4 dB at all:

| N_in | tau=4 ms | tau=8 ms | tau=20 ms |
|---:|---:|---:|---:|
| 32 | 4.80 | 5.77 | 4.09 |
| 128 | 9.30 | 8.88 | 4.65 |
| 512 | 13.48 | 10.44 | 4.86 |
| 4096 | 16.54 | 11.03 | 4.88 |

At `tau=4 ms` and N=4096 the encoder preserves **16.54 of 19.47 dB**. The apparent encoder
ceiling was an artifact of the (N=32, tau=20 ms) operating point, not a property of Poisson
rate coding. The honest statement is that **the split between encoder loss and layer loss
moves with configuration**, so naming any one stage "the bottleneck" from a single
configuration is wrong.

## 3. What actually binds

Re-running the decomposition where the readout is *not* the constraint — `tau=4 ms`,
uniform 128-wide layers — localises it:

| stage | units | population | Δ | linear (test) | clipped |
|---|---:|---:|---:|---:|---:|
| analog input | 1 | 19.47 ± 0.03 | — | 19.39 ± 0.18 | 0.00% |
| encoded | 128 | 8.56 ± 0.06 | −10.91 | 8.25 ± 0.15 | 12.51% |
| **layer1** | 128 | **1.90 ± 0.02** | **−6.66** | 1.29 ± 0.06 | 16.69% |
| layer2 | 128 | 0.46 ± 0.03 | −1.43 | 0.01 ± 0.01 | 16.31% |
| layer3 | 128 | 0.37 ± 0.03 | −0.10 | 0.17 ± 0.11 | 16.69% |

**Each LIF layer retains roughly 22% of the information reaching it.** Layer 1 takes 8.56 dB
down to 1.90; layer 2 takes 1.90 to 0.46; layer 3 adds almost nothing because there is
almost nothing left. After two LIF layers the **optimal linear readout is 0.01 dB** — no
linear decoder can recover the clean signal from that representation at all.

Three subsidiary findings:

- **It is not a decoder failure.** Population readout and optimal linear readout agree
  closely at every stage in every configuration. The information is genuinely destroyed,
  not merely inaccessible to the population mean. A better decoder will not fix this.
- **Clipping is a real but secondary contributor.** `PopulationDecoder` clips normalised
  rate to [0,1] (`population.py:61`): 12–17% of samples are discarded at `tau=4 ms`, but
  only 0.1–4% at the shipped `tau=20 ms`.
- **Width change is not the cause of the per-layer drop.** The `--uniform` control
  ([32,32,32], so width never changes between stages) gives per-layer losses of −0.75 and
  −0.67, against −0.75 and −0.61 for the shrinking default. The loss is intrinsic to the
  LIF transformation, not to having fewer neurons to average.

## 4. Why every knob looked invariant

The loss **redistributes** between encoder and layers as configuration changes, while the
total stays far more stable than any single stage:

| configuration | encoder | after LIF layers | output |
|---|---:|---:|---:|
| `[32,16,8]`, tau=20 ms | 3.74 | 2.48 → 1.33 | **1.33** |
| `[128,128,128]`, tau=4 ms | 8.56 | 1.90 → 0.46 | **0.46** |

Shortening `tau` buys 4.8 dB at the encoder and loses it again in the layers. This is why
Findings 1–5 and M1a all reported flat responses: the pipeline has compensating losses, so
single-knob sweeps move information between stages rather than out of the system.

## 5. What the architecture can actually do

Since depth is the dominant sink, the shallowest network should be strongest. Full
pipeline, input 20 dB:

| layers | tau=4 | tau=8 | tau=12 | tau=20 | tau=40 |
|---|---:|---:|---:|---:|---:|
| `[128,128]` (1 LIF layer) | 1.90 | 2.41 | 2.85 | **3.43** | 2.27 |
| `[512,512]` (1 LIF layer) | 2.10 | 2.70 | 3.21 | **3.77** | 2.41 |
| `[128,128,128]` (2 layers) | 0.46 | 0.69 | 0.91 | 1.50 | 1.40 |
| `[32,16,8]` (shipped) | 0.45 | 0.66 | 0.85 | **1.33** | 1.26 |

Three design rules, in order of effect size:

1. **Depth dominates.** Dropping from two LIF layers to one is worth **+1.93 dB**
   (1.50 → 3.43 at width 128), far more than any width or tau change.
2. **`tau_decode = 20 ms` is already optimal** for the full pipeline, in every
   configuration tested. The shipped value is right, and Finding 2's non-monotonic τ
   response peaking near 20 ms is reproduced here.
3. **Width helps modestly once depth is minimal** — 128 → 512 is worth +0.34 dB.

Best configuration found: `[512,512]`, tau=20 ms → **3.77 dB**, against the shipped
default's 1.33 dB. So the ceiling is roughly **2.8× higher than the default suggests**, and
still loses ~80% of a 19.47 dB input.

## 6. Consequences

**For the paper.** The claim is no longer "rate coding has a hard ceiling." It is:
*in a rate-coded unsupervised-STDP reconstruction pipeline the dominant information sink is
the LIF layer itself, which retains ~22% of what reaches it; losses redistribute across
stages under configuration change, which is why the end-to-end ceiling appears invariant to
every individual architectural knob.* That is a sharper and better-supported claim, and it
comes with an actionable design rule (minimise depth).

**For M3.** The target has changed. Deriving the Poisson encoder's capacity bound is no
longer the priority, because the encoder is not what binds. The quantity to derive is the
**per-layer information retention of a calibrated LIF population under rate coding** — why
~22%, and what it depends on.

**Caveats.**
- Passive networks only. Training is untested here; Finding 2 put it at 0.11–0.44 dB, but
  that was at fixed global gain.
- Synthetic ECG, 3 seeds, 20 dB input. Real MIT-BIH and proper power are still owed.
- The ~22% retention figure is measured at one operating point (tau=4 ms, width 128,
  40 Hz target rate). It should be mapped across firing rate and gain before being stated
  as a law in the manuscript — at present it is a measurement, not a result.
- The linear readout is fit with up to 128 features on ~1000 training samples; the
  train/test gap (e.g. 10.33 vs 8.25 at the encoder) shows mild overfitting, so the test
  column is the one to quote.
