# ICEE 2026 paper

IEEE ICEE 2026 (8th International Conference on Emerging Electronics, Bengaluru), track
*Memory Technologies & Neuromorphic Computing*. Limit: 4 pages including references.

## Build

```bash
uv run python scripts/paper_icee2026/figures.py   # Figs. 2-3 and Table I, from committed analyses
cd paper/icee2026 && latexmk -pdf -outdir=build main.tex
```

`build/main.pdf` is the submission file. Fig. 1 (the pipeline) is TikZ inside `main.tex`.
`figures/` is generated; do not edit it by hand.

## Where the numbers come from

| Paper | Source |
|---|---|
| Fig. 2, Section IV-A | `docs/r3_findings.md` §1.1–§1.5; `results/r3_characterisation/analysis/c1_*.csv` |
| Table I, Section IV-B | R3 C2 and C3 (`c2_comparisons.csv`, `c3_rate_comparisons.csv`), R5 T1 (`r5_t1_encoders.csv`), R4b §3.2 (`r4b_fan_in_test.csv`) |
| Fig. 3, Section IV-C | `docs/r4b_findings.md` §1 and §3.2; `docs/m4_findings.md` §3 |
| Discussion | R4b §2 (encoder range), `docs/m1a_findings.md` (independent input), R5 §1.5 (delay) and §2.3 (QRS filter baseline) |
| Section IV-A, "Independent noise, and what sets the height"; IV-B per-record 5 Hz split (camera-ready, post hoc) | `scripts/paper_icee2026/icee_decomposition.py` → `results/icee2026_revision/` (`decomposition_summary.csv`, `clip_check.csv`, `rate_by_record.csv`) |
| Section IV-A, sine law RMS 0.18–0.23 dB | `docs/r3_findings.md` §1.5 table |

The table generator refuses to write Table I if any row's grouping (does not raise /
lowers / raises) disagrees with its test result.

## Submission

Submitted to EDAS on 14 September 2026 as a student paper, with the PDF certified by IEEE
PDF eXpress (conference ID 72661X). Decision due 14 October 2026.

Format decisions that the checks depend on:

- Page size is US Letter, which is what ICEE's "8-1/2 x 11 in" requirement means.
- Captions are 9 pt, as ICEE requires: `main.tex` redefines IEEEtran's caption macro with
  `\small` in place of `\footnotesize` (8 pt). The class file is unmodified.
- `microtype` runs with `protrusion=false`. Protrusion pushed line-end punctuation into
  the column gutter, which EDAS flagged at 0.115 in against its 0.12 in minimum; without
  it the gutter is 0.166 in on every page.

Any change to the PDF, including the camera-ready, needs a fresh PDF eXpress certificate
before it is uploaded to EDAS.

## Camera-ready (October 2026)

Accepted as a poster, with the full paper invited for IEEE Xplore. Final paper due
21 October 2026: 4 pages including references, IEEE PDF eXpress (72661X), IEEE
e-Copyright, then upload at edas.info/N35445. Presenting-author registration closes
25 October 2026.

How the reviews were addressed (no new simulations; one post hoc analysis of committed
data, `icee_decomposition.py`):

| Request | Change |
|---|---|
| R1: qualify "no fitted parameters" | Abstract, contributions, §III, conclusion: parameter-free *given* the ceiling from one noiseless run; *C* is measured, not derived from the architecture |
| R1: validate the independent additive-noise assumption | §IV-A: closed-form Bernoulli shot noise predicts the measured encoder stage to 0.11 dB RMS over 16 records; the EEG/sine excess is the readout clip (`clip_check.csv`) |
| R1: copied-input vs. internal correlation | §III: at noiseless input the encoders are independent, so the shared error is made by the network; M1a independent-noise result moved here |
| R1: change only fan-in | §IV-C states that depth, width, code and rate are held fixed (as the R4b grid was run) |
| R1: fan-in rule is configuration-specific; scope; conventional filters win | Abstract, contributions, Discussion ("for the networks and signals studied here"), Limitations, conclusion |
| R1: why the ceiling differs across signals; separate spike noise, causal readout and filtering | §IV-A decomposition: readout filter (and its delay), encoder spikes and LIF layers as shares of 1/C; coefficient of variation on the encoder range |
| R1: the 5 Hz improvement | §IV-B: event-detector account, per-record split (`rate_by_record.csv`), and the EEG/sine contrast |
| R1, R2: STDP causality; copied input | §IV-C: fixing K is the causal intervention; STDP-through-fan-in is correlational and untested on non-redundant inputs |
| R1: other datasets; recurrent / inhibitory / trained readouts | Limitations: conclusions limited to the class simulated and one real database |
| R2: cite Zohary 1994, Diesmann 1999, van Rossum 2002, Litwin-Kumar 2017 | Introduction and §III/§IV-C; refs checked against Crossref, an author must confirm the cited statements |
| R2: higher-fidelity regime | §IV-A: the sine is the highest-fidelity test (ceiling S/N 4.6) and fits less well (RMS 0.18–0.23 dB); Limitations: untested near the delay-compensated limit |
| R2: code release | §II-D: repository URL |
| R2: Table I mixes baselines | Table I gains a Ref. column (A–D) defined in its notes (`figures.py`) |

To make room: the design-rule bullets became a paragraph, the stage equation went
inline, the record list moved to the code, Fig. 1's duplicate equation was removed, and
references with more than six authors use "et al." (IEEE style).
