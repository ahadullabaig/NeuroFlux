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
