# MCSoC 2027 paper

*One Layer, Eight Inputs, Two Bits: The Price of Fidelity in Spiking Biosignal Processors.*

The venue is the 20th IEEE MCSoC 2027, NeuroCore Global Summit (UEC Tokyo, 12–15 May 2027).
It takes regular papers of up to 8 pages including references; we are staying at 8.
Submission is through EDAS N35724 by 30 November 2026, and review is single-blind.

## Build

```bash
uv run python scripts/h1_hardware_cost/break_even.py   # break-even tables (descriptive)
uv run python scripts/paper_mcsoc2027/mcsoc_figures.py       # Figs. 2-6 and Tables I-III
cd paper/mcsoc2027 && latexmk -pdf -outdir=build main.tex
```

`build/main.pdf` is the draft. Fig. 1 is TikZ inside `main.tex`. `figures/` is generated; do not
edit it by hand.

## Where the numbers come from

All sources are in `docs/h1_findings.md`, with data in `results/h1_hardware_cost/analysis/`.

| Paper | Source |
|---|---|
| Fig. 2, Table I, §V-A | §1 (pilot B: `pilot_b_*.csv`) and §3.2 (`b2_front.csv`) |
| Fig. 3, §V-B | §3.1 and §3.3 (`b2_tests.csv`, `b2_cells.csv`, `b2_post_hoc.csv`, `b2_p2b_cells.csv`) |
| Table II, Figs. 4–5, §V-C | §7 (`confirm_tests.csv`, `confirm_cells.csv`, which carries the static costs) and §8 (`break_even_*.csv`) |
| Fig. 6, Table III, §V-D | §6 (`pilot_a_tests.csv`, `pilot_a_cells.csv`, `pilot_a_post_hoc.csv`) |
| §VI, mapping to hardware | Counts from Table II and §III's cost model; no new measurement |
| Background: ceiling, cascade law, fan-in rule | our ICEE 2026 paper (`paper/icee2026/`), cited, with no text reused |

Pre-registration commits, each made before its grid ran:
- pool grid (B2): the rules are in `scripts/h1_hardware_cost/b2_grid.py`, committed in ee87931
  together with the grid's results, so there is no separate pre-registration commit; the
  docstring records the one amendment made mid-run;
- device grid: c220dac;
- confirmation grid: 19766c5.

## Before submission

- [ ] **Read every `[MUST READ]` entry in `refs.bib`** and confirm the sentence that cites it,
      or remove the citation. The entries are Aliyev 2024, Sorbaro 2020, Chee 2025, Boahen 2000,
      Ganesh 2025, Sulistiyanto 2025, Yu 2018, Chen 2017, Rathi 2019 and Putra 2021.
- [ ] **ICEE citation (`baig2026icee`).** After 14 Oct 2026, change it to "to appear" if ICEE
      accepts. If ICEE rejects, remove it and state the ceiling, the law and the fan-in rule
      directly. Declare the ICEE paper as related prior work in EDAS.
- [x] **Overlap check.** No sentence may be copied from `paper/icee2026/main.tex`.
      `uv run python scripts/paper_mcsoc2027/overlap.py` reports no shared run of six or more
      words, no sentence with trigram overlap ≥ 0.3, and no shared figure file. The first run
      found 19 runs in the Methods, now rewritten. Re-run after every edit.
- [x] **Number audit.** Every number in `main.tex` must match an analysis CSV.
      `uv run python scripts/paper_mcsoc2027/audit.py` gives 124 checks passed and 0 failed,
      plus a list of sentences with numbers no check covers, for a human read. Re-run after
      every edit.
- [ ] **Page budget.** The draft is 6 of 8 pages with the four additions (Fig. 3c, Table II
      hardware columns, Fig. 5 noisy inputs, the mapping-to-hardware subsection). Use the rest
      for substance, not padding.
- [ ] **Format.** US Letter, template 8 pt captions, fonts embedded, column gutter ≥ 0.12 in
      (microtype protrusion is already off), and PDF eXpress at camera-ready. Body text leaves a
      0.164–0.166 in gutter on every page. On page 3, the full-width Fig. 2 and Table II place
      text in the gutter by design; if EDAS's gutter check counts them, move those floats.
- [ ] **EDAS.** Choose the NeuroCore track; all three authors need EDAS profiles.
