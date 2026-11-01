# Documentation

This folder is the project's lab notebook. Each findings document records one stage: the question, the analysis fixed before the data came in, the data, the result, and any later corrections.

## Start here

1. The [README](../README.md) gives the overview and the headline results.
2. [`paper/icee2026/`](../paper/icee2026/) holds the accepted paper and a map from every number in it to its source.
3. [R3](r3_findings.md), [R4b](r4b_findings.md) and [R5](r5_findings.md) are the evidence behind the paper. Read them in that order.

## Stage names

- **Phase 0–3** are stages of the [original plan](archive/PLAN.md). Phase 3 was the first sweep on real ECG.
- **R0–R6** are stages of the [paper roadmap](paper_roadmap.md), which replaced the plan after Phase 3: R0 literature, R1 measurement validity, R2 statistical validity, R3 ceiling characterisation, R4 mechanism (sub-stages M1a–M4, then R4b), R5 generality and R6 the manuscript.
- **H1** studies hardware cost, for the follow-up paper.

Each stage's code is in `scripts/<dir>/` and its outputs are in `results/<dir>/` ([index](../results/README.md)).

## Findings, in order

| Stage | Question | Outcome | `<dir>` |
|---|---|---|---|
| [Phase 3](phase3_findings.md) | Does NeuroFlux denoise real ECG? | No. It reconstructs at a fixed quality. Its R-peak claims were later reversed by R1 and R2. | `phase3_diagnostics` |
| [R0](related_work.md) | Is the ceiling already known? | Literature review. It gave the paper its positioning and citations. | — |
| [R1](r1_findings.md) | Does the clinical metric measure anything? | No. `rpeak_integrity` was saturated, and QRS F1 scored against annotations replaced it. | `r1_metrics` |
| [R2](r2_findings.md) | Were the Phase 3 statistics valid? | No. Records were pooled as if they were seeds. Every claim was re-tested with the record as the unit. | `r2_reanalysis` |
| [M1a](m1a_findings.md) | Is the ceiling an artifact of copying one signal to every input? | No. Independent input channels do not lift it. | `m1a_shared_input` |
| [M2](m2_findings.md) | Where is the fidelity lost? | Mostly in the LIF layers, not in the encoder or the readout. | `m2_bottleneck` |
| [M3](m3_findings.md) | Can the ceiling be derived? | Yes. Inverse S/N ratios add from stage to stage (`metrics/cascade.py`). | `m3_bound` |
| [M4](m4_findings.md) | Does sparse initialisation help by decorrelating or by cutting fan-in? | By decorrelating. Fan-in sets the optimum. | `m4_decorrelation` |
| [R3](r3_findings.md) | How flat is the ceiling, what moves it, and where does the network operate? | Flat on every signal. The law is exact at the operating point. Output rate is the largest operating-point lever. | `r3_characterisation` |
| [R4b](r4b_findings.md) | Does the mechanism hold at R3's operating points? | STDP collapses effective fan-in. A fixed fan-in of 16 reaches 5.77 dB. | `r4b_mechanism` |
| [R5](r5_findings.md) | Does the ceiling generalise across encoders, readouts and QRS detectors? | Flat under three codes. Every earlier ceiling was causal. No network beats a cleaned input. | `r5_generality` |
| [H1](h1_findings.md) | What does fidelity cost in hardware? | Ongoing. Low-precision weights and device variation are tolerated. A wide input pool behind a fixed fan-in is the cost lever. | `h1_hardware_cost` |

The camera-ready analysis for the ICEE paper is in `scripts/paper_icee2026/` and `results/icee2026_revision/`.

## Reading a findings document

- The **Status** line at the top says whether the stage is complete. The **Reproduce** block lists the exact commands.
- **Correction blocks** (quoted, near the top) override the text below them. Earlier text is left in place so the record stays honest.
- **Post hoc** marks any analysis designed after seeing the data.
- **Ceilings are causal** unless a document says *delay-compensated*. **Input SNRs in Phase 3, R1 and R2 are nominal.** Flicker was added before the noise, so nominal 0 dB is really −0.9 dB.

## Other documents

- [`paper_roadmap.md`](paper_roadmap.md): the plan that produced the paper, and why the denoising framing was dropped.
- [`related_work.md`](related_work.md): the literature, with what each paper is cited for.
- [`archive/`](archive/): the original proposal and 16-week plan, kept for the record.
