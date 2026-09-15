"""
Number audit for the MCSoC 2027 manuscript (`paper/mcsoc2027/main.tex`).

Every number in the text is checked against its source: a committed H1
analysis in `results/h1_hardware_cost/analysis/`, a grid script's constants, a
raw grid's row count, or the simulator's own defaults. A claim names a passage
of the manuscript, recomputes the value, renders it at the paper's precision,
and fails if that rendering is not in the passage. Passages are matched after
LaTeX markup is stripped, so `$-0.23$ [$-0.45$, $+0.02$]\\dB` matches
"-0.23 [-0.45, +0.02] dB".

The tables are generated from the same analyses by `mcsoc_figures.py`; the
audit regenerates them into a temporary directory and fails if they differ from
the committed ones.

A coverage report follows: every manuscript line carrying a number that no
claim covered, so a number added later cannot slip past unaudited.

Usage:
    uv run python scripts/paper_mcsoc2027/audit.py
Exit status is non-zero if any claim fails.
"""

from __future__ import annotations

import csv
import gzip
import math
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "r3_characterisation"))
sys.path.insert(0, str(ROOT / "scripts" / "h1_hardware_cost"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_mcsoc2027"))

import b2_grid  # noqa: E402
import confirm_grid  # noqa: E402
import mcsoc_figures  # noqa: E402
import pilot_a_grid  # noqa: E402
import r3_common  # noqa: E402
from neuroflux_core.io.config import LIFConfig  # noqa: E402
from neuroflux_core.metrics.cascade import (  # noqa: E402
    affine_to_excess_db,
    db_to_linear,
)
from neuroflux_datasets.signals import PHASE3_RECORDS  # noqa: E402

TEX = ROOT / "paper" / "mcsoc2027" / "main.tex"
ANALYSIS = ROOT / "results" / "h1_hardware_cost" / "analysis"
RESULTS = ROOT / "results" / "h1_hardware_cost"

Row = dict[str, str]


def load(name: str) -> list[Row]:
    with (ANALYSIS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grid_rows(name: str) -> list[Row]:
    with gzip.open(RESULTS / name, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalise(text: str, source: bool = False) -> str:
    """
    LaTeX to plain words: strip markup that sits between numbers and units.

    `source` marks manuscript text, whose unescaped `%` starts a comment; a
    claim's passage is plain text, where `%` is a percent sign.
    """
    if source:
        text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace("\\dB{}", " dB").replace("\\dB", " dB")
    text = re.sub(r"\\(ref|eqref|cite|label)\{[^}]*\}", "", text)
    text = text.replace("\\{", "").replace("\\}", "")
    text = text.replace("~", " ").replace("$", "").replace("\\%", "%")
    text = text.replace("{", "").replace("}", "").replace("--", "-")
    text = re.sub(r"\\times10\^(-?\d+)", r"e\1", text)
    text = re.sub(r"\s+", " ", text)
    return text


def f2(x: float) -> str:
    return f"{x:.2f}"


def signed(x: float, digits: int = 2) -> str:
    if round(x, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{x:+.{digits}f}"


class Audit:
    def __init__(self, tex: str) -> None:
        self.tex = normalise(tex, source=True)
        self.failures: list[str] = []
        self.passed = 0
        self.covered: list[str] = []

    def claim(self, passage: str, *expected: str, label: str = "") -> None:
        """`passage` must be in the manuscript and contain every `expected`."""
        text = normalise(passage)
        self.covered.append(text)
        name = label or text[:70]
        if text not in self.tex:
            self.failures.append(f"PASSAGE NOT FOUND: {name!r}")
            return
        missing = [e for e in expected if e not in text]
        if missing:
            self.failures.append(f"MISMATCH in {name!r}: expected {missing}")
        else:
            self.passed += 1

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(f"FAILED {label}: {detail}")


def interval(row: Row) -> tuple[str, str, str]:
    return (
        signed(float(row["median_diff"])),
        signed(float(row["ci_low"])),
        signed(float(row["ci_high"])),
    )


def run(a: Audit) -> None:
    contrasts = {r["contrast"]: r for r in load("pilot_b_contrasts.csv")}
    points = [r for r in load("pilot_b_points.csv") if r["signal"] == "mit-bih"]
    b2_tests = {r["test"]: r for r in load("b2_tests.csv")}
    b2_cells = load("b2_cells.csv")
    b2_post = load("b2_post_hoc.csv")
    p2b_cells = load("b2_p2b_cells.csv")
    pa_tests = load("pilot_a_tests.csv")
    pa_cells = load("pilot_a_cells.csv")
    pa_post = {r["group"]: r for r in load("pilot_a_post_hoc.csv")}
    c_tests = {r["test"]: r for r in load("confirm_tests.csv")}
    c_cells = {
        (r["signal"], r["input_snr"], r["design"]): r for r in load("confirm_cells.csv")
    }
    switches = load("break_even_switches.csv")
    crossings = {
        (r["design_a"], r["design_b"]): r
        for r in load("break_even_designs.csv")
        if r.get("design_a")
    }

    def cell(signal: str, snr: str, design: str, key: str = "ceiling_db") -> float:
        return float(c_cells[(signal, snr, design)][key])

    depth = contrasts["depth 2 vs 1 LIF layer"]
    width = contrasts["input width 256 vs 32, output 40 Hz"]
    k16 = contrasts["[512,512] K=16 vs [128,128] K=8, 20 Hz"]
    rate = contrasts["output rate 5 Hz vs 40 Hz"]
    p2b = b2_tests["P2b law predicts 10 Hz from 20 and 40 Hz, all cells"]
    f1 = b2_tests["F1 [2048,128] minus [128,128], 40 Hz"]
    p1_32 = b2_tests["P1 slope on log2(N_in), N_1 = 32"]
    p1_128 = b2_tests["P1 slope on log2(N_in), N_1 = 128"]
    p2a = [r for t, r in b2_tests.items() if t.startswith("P2a")]
    c1 = c_tests["C1 headline, D2 vs D1"]
    c2 = c_tests["C2 two bits, D3 vs D2"]
    c3 = c_tests["C3 break-even pair, D4 vs D5 (non-inferiority)"]
    c4 = c_tests["C4 costs"]
    c5_10 = c_tests["C5 D2 vs D1 at 10 dB input"]
    c5_0 = c_tests["C5 D2 vs D1 at 0 dB input"]
    g1_eeg = c_tests["G1 D2 vs D1 on eeg"]
    g1_sine = c_tests["G1 D2 vs D1 on sine"]

    def level_changes(level: str) -> list[float]:
        return [float(r["change_db"]) for r in pa_cells if r["level"] == level]

    worst = {lv: min(level_changes(lv)) for lv in {r["level"] for r in pa_cells}}
    perturbed = [r for r in pa_cells if r["level"] != "exact"]
    n_design_points = len(points) + len(b2_cells) + len(perturbed)
    reference = next(r for r in points if r["design"] == "[32,16,8] @40;40")

    def excess_ratio(high_db: float, low_db: float) -> float:
        return db_to_linear(affine_to_excess_db(high_db)) / db_to_linear(
            affine_to_excess_db(low_db)
        )

    # ------------------------------------------------------------ abstract
    a.check(
        "about 200 design points", 190 <= n_design_points <= 210, f"{n_design_points}"
    )
    a.claim("attach a hardware cost to about 200 design points")
    a.claim(
        "a second layer costs 2.5 times the synaptic work and 0.9 dB of fidelity",
        f"{float(depth['synops_per_s_ratio']):.1f}",
        f"{abs(float(depth['median_diff'])):.1f}",
    )
    a.claim(
        "predicts the fidelity cost of a slower encoder to within 0.07 dB",
        f"{float(p2b['median_abs_error_db']):.2f}",
    )
    a.check(
        "D2 is [512,128] K=8",
        confirm_grid.DESIGNS["D2"][0] == (512, 128)
        and confirm_grid.DESIGNS["D2"][3] == 8,
    )
    a.claim("One layer of 128 neurons, each reading 8 of 512 input channels")
    a.check(
        "one eighth of synops and synapses, three quarters of spikes",
        f2(float(c4["d2_synops_ratio_to_d1"])) in ("0.12", "0.13")
        and cell("mit-bih", "inf", "D2", "synapses")
        / cell("mit-bih", "inf", "D1", "synapses")
        == 0.125
        and f2(float(c4["d2_spikes_ratio_to_d1"])) == "0.75",
    )
    a.claim(
        "with one eighth of its synaptic operations and synapses and three "
        "quarters of its spikes"
    )
    a.claim("loses at most 0.22 dB with 1-bit weights", f"{abs(worst['1 bit']):.2f}")
    a.claim(
        "loses at most 0.69 dB when three quarters of its synaptic devices are "
        "stuck at zero",
        f"{abs(worst['stuck 0.75']):.2f}",
    )
    a.check(
        "C2 equivalent (keeps fidelity with 2-bit weights)", c2["equivalent"] == "True"
    )

    # ------------------------------------------------------------ introduction
    a.claim(
        "Past that choice, fidelity costs about 0.2 dB per doubling of synaptic work"
    )
    a.check(
        "about 0.2 dB per doubling (width and wider fan-in layer)",
        all(round(float(r["db_per_synops_doubling"]), 1) == 0.2 for r in (width, k16)),
    )
    a.claim(
        "raises the ceiling by 0.2-0.3 dB per doubling",
        f"{float(p1_32['median_diff']):.1f}",
        f"{float(p1_128['median_diff']):.1f}",
    )
    a.claim(
        "prices the encoder rate to 0.07 dB", f"{float(p2b['median_abs_error_db']):.2f}"
    )
    a.claim(
        "weights cost at most 0.22 dB, and a quarter of devices stuck at zero at "
        "most 0.15 dB",
        f"{abs(worst['1 bit']):.2f}",
        f"{abs(worst['stuck 0.25']):.2f}",
    )
    slope = float(pa_post["all"]["slope_through_origin"])
    a.check("overstates about twofold", 0.4 <= slope <= 0.6, f"slope {slope}")

    # ------------------------------------------------------------ model and method
    a.check("250 Hz, 4 ms steps", r3_common.FS == 250.0 and r3_common.DT == 4.0)
    a.claim("A signal sampled at 250 Hz (4 ms steps)")
    lif = LIFConfig()
    a.claim(
        "a resting potential of -70 mV, a -55 mV threshold, a -80 mV reset, a 20 ms "
        "membrane time constant and a 2 ms refractory period",
        f"{lif.v_rest:.0f}",
        f"{lif.v_thresh:.0f}",
        f"{lif.v_reset:.0f}",
        f"{lif.tau_m:.0f} ms membrane time constant",
        f"{lif.t_ref:.0f} ms refractory period",
    )
    a.check(
        "readout time constant 20 ms",
        r3_common.Spec("x", "x", "x", 0).decoder_taus == (20.0,),
    )
    a.claim("exponential filter of 20 ms time constant")
    fan_in_src = (ROOT / "scripts/r3_characterisation/r3_common.py").read_text()
    a.check(
        "fixed fan-in weights uniform on [0,1]",
        "rng.uniform(0.0, 1.0, size=fan_in)" in fan_in_src,
    )
    a.claim(
        "with weights uniform on [0,1]; dense layers read every channel, with "
        "weights uniform on [0, 0.3]"
    )
    topo = (ROOT / "packages/core/src/neuroflux_core/network/topology.py").read_text()
    a.check("dense (scaled init) weights on [0, 0.3]", "scale: float = 0.3" in topo)
    a.claim("updates every neuron at every 4 ms step, 250 \\sum_k S_k reads and 250")
    a.check("16 records", len(PHASE3_RECORDS) == 16)
    a.claim("The ECG data are 16 records from the MIT-BIH Arrhythmia Database")
    a.check("10 s windows", r3_common.DURATION_S == 10.0)
    a.claim(
        "picks another 10 s segment free of artifact annotations, resampled to 250 Hz"
    )
    synth = (ROOT / "packages/datasets/src/neuroflux_datasets/synthetic.py").read_text()
    a.check(
        "EEG bands 10, 20, 40 Hz; 2 Hz sine",
        "bands = [(10.0, 1.0), (20.0, 0.5), (40.0, 0.25)]" in synth
        and "freq: float = 2.0" in synth,
    )
    a.claim("randomly phased 10, 20 and 40 Hz sinusoids, and a sine wave at 2 Hz")
    a.claim("The landscape combines 76 MIT-BIH design points", str(len(points)))
    a.check(
        "pool grid constants",
        b2_grid.FAN_IN == 8
        and b2_grid.LAYER_WIDTHS == (32, 128)
        and b2_grid.INPUT_WIDTHS == (32, 128, 512, 2048)
        and b2_grid.ENCODER_RATES_HZ == (10.0, 20.0, 40.0)
        and b2_grid.LAYER_RATE_HZ == 20.0
        and len(b2_cells) == 24,
    )
    a.claim(
        "pool grid of 24 one-layer designs: K = 8, N_1 \\in 32, 128, "
        "\\Nin \\in 32, 128, 512, 2048, r_0 \\in 10, 20, 40 Hz and r_1 = 20 Hz"
    )
    a.claim("(1,920 runs)", f"{len(grid_rows('b2_grid.csv.gz')):,}")
    a.check(
        "device grid pools and rates",
        pilot_a_grid.INPUT_WIDTHS == (128, 512, 2048)
        and pilot_a_grid.LAYER_WIDTH == 128
        and pilot_a_grid.ENCODER_RATES_HZ == (10.0, 20.0, 40.0),
    )
    a.claim("(8,640 runs)", f"{len(grid_rows('pilot_a_grid.csv.gz')):,}")
    a.check(
        "six named designs, seeds 10-19, inputs",
        len(confirm_grid.DESIGNS) == 6
        and confirm_grid.SEEDS == tuple(range(10, 20))
        and confirm_grid.INPUT_SNRS_DB == (math.inf, 10.0, 0.0),
    )
    a.claim("on seeds 10-19 (3,240", f"{len(grid_rows('confirm_grid.csv.gz')):,}")
    a.claim("at least 103 clean windows")
    try:
        from neuroflux_datasets.mit_bih.annotator import extract_annotated_segments

        counts = [
            len(extract_annotated_segments(r, "data/mit_bih", 10.0, fs=250.0))
            for r in PHASE3_RECORDS
        ]
        a.check("at least 103 clean windows", min(counts) == 103, f"min {min(counts)}")
    except Exception as exc:  # noqa: BLE001 — records not cached here
        print(f"  (skipped the window count: {exc})")
    a.check(
        "pool and device grids used seeds 0-4",
        {r["seed"] for r in grid_rows("b2_grid.csv.gz")} == {str(s) for s in range(5)}
        and {r["seed"] for r in grid_rows("pilot_a_grid.csv.gz")}
        == {str(s) for s in range(5)},
    )
    reproduced = [r for r in b2_tests.values() if r["test"].startswith("check")] + [
        r for r in pa_tests if r["test"].startswith("check")
    ]
    a.check(
        "reproduction checks 80 and 720 exact",
        sorted(int(r["n_matched"]) for r in reproduced) == [80, 720]
        and all(r["verdict"] == "reproduced" for r in reproduced),
    )
    a.claim("reproduce exactly (80 and 720 runs)")

    a.claim(
        "matches the previous best design within \\pm0.5 dB at one eighth of the "
        "synaptic operations, with 2-bit weights, on unseen ECG and at noisy inputs, "
        "and is no worse on two further signals"
    )
    a.check(
        "intro bullet: C1, C2, C5 equivalent; G1 non-inferior",
        all(r["equivalent"] == "True" for r in (c1, c2, c5_10, c5_0))
        and all(float(r["ci_low"]) > -0.5 for r in (g1_eeg, g1_sine)),
    )
    import analyse_r3  # noqa: PLC0415
    from neuroflux_datasets import analysis as stats_lib  # noqa: PLC0415

    earlier_seeds = [
        {r["seed"] for r in _rows(ROOT / p)}
        for p in (
            "results/r4b_mechanism/fan_in.csv.gz",
            "results/r4b_mechanism/width.csv.gz",
            "results/r3_characterisation/c3_rate.csv.gz",
        )
    ]
    a.check(
        "earlier grids used 10 seeds",
        all(seeds == {str(x) for x in range(10)} for seeds in earlier_seeds),
    )
    a.claim("Earlier grids used 10 seeds, and the pool and device grids used seeds 0-4")
    a.claim(
        "Records (n = 16) are the statistical units on ECG and seeds (n = 10) on the "
        "synthetic signals"
    )
    a.check(
        "95% BCa interval, TOST at 0.5 dB",
        analyse_r3.SNR_MARGIN == 0.5
        and stats_lib.bootstrap_median_ci.__defaults__ is not None
        and 0.95 in stats_lib.bootstrap_median_ci.__defaults__,
    )
    a.claim("95% bias-corrected and accelerated (BCa) bootstrap interval")
    a.claim("two one-sided tests reject a difference of \\pm0.5 dB")
    a.claim("It uses noiseless, 10 dB and 0 dB inputs on MIT-BIH, EEG and the sine")
    higher = [
        (s, d)
        for s in ("eeg", "sine")
        for d in confirm_grid.DESIGNS
        if cell(s, "10.0", d) > cell(s, "inf", d)
    ]
    a.check(
        "some EEG designs and D1 on the sine score higher at 10 dB than noiseless",
        any(s == "eeg" for s, _ in higher)
        and [d for s, d in higher if s == "sine"] == ["D1"],
        f"{higher}",
    )
    a.claim(
        "Some designs on EEG, and D1 on the sine, also score slightly higher at 10 dB "
        "input than at a noiseless one"
    )

    # ------------------------------------------------------------ landscape
    a.claim("100 exact-weight designs on MIT-BIH", str(len(points) + len(b2_cells)))
    a.claim(
        "reaches 1.92 dB for 2.6\\times10^4 synaptic operations per second",
        f2(float(reference["ceiling_db"])),
        f"{float(reference['synops_per_s']) / 1e4:.1f}",
    )
    a.claim(
        "removing its hidden layer gains 0.91 dB and cuts synaptic work by a factor "
        "of 2.5",
        f"{abs(float(depth['median_diff'])):.2f}",
        f"{float(depth['synops_per_s_ratio']):.1f}",
    )
    a.claim(
        "eightfold buys 0.45 dB for 6.6 times the synaptic work, about 0.2 dB per",
        f2(float(width["median_diff"])),
        f"{float(width['synops_per_s_ratio']):.1f}",
        f"{float(width['db_per_synops_doubling']):.1f}",
    )
    a.claim(
        "buys the same rate, 0.21 dB per doubling",
        f2(float(k16["db_per_synops_doubling"])),
    )
    a.claim(
        "operating-point setting (+0.65 dB at 5 Hz)", signed(float(rate["median_diff"]))
    )
    a.claim(
        "account for 80% of the synaptic operations",
        f"{100 * float(reference['input_synops_share']):.0f}",
    )
    front = [r for r in load("b2_front.csv")]
    one_layer = all(r["design"].count(",") == 1 for r in front)
    a.check("every point on both fronts is one layer", one_layer)

    # ------------------------------------------------------------ pool and encoder
    a.claim("by +0.20 [+0.12, +0.29] dB per doubling at N_1 = 32", *interval(p1_32))
    a.claim("and +0.29 [+0.15, +0.42] at N_1 = 128", *interval(p1_128))
    a.check(
        "Holm p = 6e-5 for both P1 slopes",
        all(f"{float(r['p_holm']):.0e}" == "6e-05" for r in (p1_32, p1_128)),
    )
    a.claim("it gains +0.57 [+0.28, +0.90] dB", *interval(f1))
    halving = [float(r["db_per_halving"]) for r in p2a]
    a.claim("costs 0.34-0.80 dB per halving", f2(min(halving)), f2(max(halving)))
    a.check(
        "P2a significant at all eight",
        len(p2a) == 8 and all(r["significant"] == "True" for r in p2a),
    )
    measured = [float(r["measured_10hz_db"]) for r in p2b_cells]
    a.claim(
        "absolute error of 0.067 dB (90th percentile 0.20 dB) over 128 "
        "record-architecture cells spanning 0.7 to 8.0 dB",
        f"{float(p2b['median_abs_error_db']):.3f}",
        f2(float(p2b["p90_abs_error_db"])),
        str(len(p2b_cells)),
        f"{min(measured):.1f}",
        f"{max(measured):.1f}",
    )
    share = {
        r["design"]: float(r["encoder_share_median"])
        for r in b2_post
        if r["analysis"].startswith("encoder")
    }
    a.claim(
        "With N_1 = 128, encoder counting noise is 44% of the output noise with a "
        "32-channel pool and 10% with 2048 channels",
        f"{100 * share['[32,128] K=8']:.0f}%",
        f"{100 * share['[2048,128] K=8']:.0f}%",
    )
    a.claim(
        "44% and 19% with N_1 = 32",
        f"{100 * share['[32,32] K=8']:.0f}%",
        f"{100 * share['[2048,32] K=8']:.0f}%",
    )
    rest = [
        float(r["lambda_rest_affine_db"])
        for r in b2_post
        if r.get("design", "").endswith(",128] K=8")
    ]
    a.claim(
        "barely moves: 5.6 to 6.1 dB at N_1 = 128",
        f"{min(rest):.1f}",
        f"{max(rest):.1f}",
    )

    # ------------------------------------------------------------ named designs
    a.claim("D2 minus D1 is -0.23 [-0.45, +0.02] dB", *interval(c1))
    a.check("C1 about 0.2 dB lower", round(abs(float(c1["median_diff"])), 1) == 0.2)
    a.claim(
        "with 0.125 times the synaptic operations, 0.125 times the synapses and 0.75 "
        "times the spikes"
    )
    a.claim("at 10 dB input (-0.27 [-0.43, -0.08])", *interval(c5_10))
    a.claim("input (-0.22 [-0.27, -0.12])", *interval(c5_0))
    a.check(
        "C5 equivalent at both inputs",
        c5_10["equivalent"] == "True" and c5_0["equivalent"] == "True",
    )
    a.claim("EEG (-0.09 [-0.29, +0.01])", *interval(g1_eeg))
    a.claim("the sine (+0.21 [+0.16, +0.30])", *interval(g1_sine))
    a.claim("by 0.00 [-0.02, +0.01] dB (D3)", *interval(c2))
    a.claim(
        "memory is then 11.3 kbit, against 335.9 kbit for D1 at 32-bit weights",
        f"{cell('mit-bih', 'inf', 'D3', 'weight_memory_bits') / 1000:.1f}",
        f"{cell('mit-bih', 'inf', 'D1', 'weight_memory_bits') / 1000:.1f}",
    )
    a.claim(
        "at -0.06 [-0.12, +0.04] dB, with a quarter of the synaptic operations and "
        "three times the spikes",
        *interval(c3),
    )
    a.check(
        "D4 synops quarter, spikes three times",
        f2(float(c4["d4_synops_ratio_to_d5"])) == "0.25"
        and f"{float(c4['d4_spikes_ratio_to_d5']):.1f}" == "3.0",
    )
    d0, d3 = cell("mit-bih", "inf", "D0"), cell("mit-bih", "inf", "D3")
    synops_d0 = cell("mit-bih", "inf", "D0", "synops_per_s")
    synops_d3 = cell("mit-bih", "inf", "D3", "synops_per_s")
    a.claim(
        "1.72 to 5.94 dB, six times the S/N, for 1.6 times the synaptic operations",
        f2(d0),
        f2(d3),
        f"{synops_d3 / synops_d0:.1f}",
    )
    a.check(
        "six times the S/N",
        round(excess_ratio(d3, d0)) == 6,
        f"{excess_ratio(d3, d0):.2f}",
    )
    a.claim(
        "cut D1's 2.05 million synapse reads per second to 256 thousand",
        f"{cell('mit-bih', 'inf', 'D1', 'clock_synapse_reads_per_s') / 1e6:.2f}",
        f"{cell('mit-bih', 'inf', 'D2', 'clock_synapse_reads_per_s') / 1e3:.0f}",
    )
    a.claim(
        "it delivers 4.04 dB against 0.72 for the reference architecture",
        f2(cell("mit-bih", "0.0", "D2")),
        f2(cell("mit-bih", "0.0", "D0")),
    )
    gaps = [
        cell(s, n, "D1") - cell(s, n, "D2")
        for s in ("eeg", "sine")
        for n in ("10.0", "0.0")
    ]
    a.claim("D2's median falls 0.35-0.67 dB below D1's", f2(min(gaps)), f2(max(gaps)))
    a.check(
        "one-layer designs above D0 at every input and signal",
        all(
            cell(s, n, d) > cell(s, n, "D0")
            for s in ("mit-bih", "eeg", "sine")
            for n in ("inf", "10.0", "0.0")
            for d in ("D1", "D2", "D3", "D4", "D5")
        ),
    )
    rho = float(crossings[("D4", "D5")]["crossover_rho"])
    a.claim("whenever \\rho < 2.00", f2(rho))
    a.claim("below \\rho^* = 2.00", f2(rho))
    a.check(
        "D2 cheaper than D1 at every rho",
        "D2 is cheaper at every rho" in crossings[("D1", "D2")]["relation"],
    )
    five = [r for r in switches if float(r["target_db"]) == 5.0]
    a.claim(
        "the 2048-channel pool at 10 Hz below \\rho = 0.99, [512,128] at 20 Hz up to "
        "\\rho = 3.25, and [128,128] with its layer at 10 Hz above it",
        *[f2(float(r["switch_rho"])) for r in five],
    )
    a.check(
        "5 dB switch designs",
        [(r["cheapest_below"], r["cheapest_above"]) for r in five]
        == [
            ("[2048,128] K=8 enc 10 Hz", "[512,128] K=8 enc 20 Hz"),
            ("[512,128] K=8 enc 20 Hz", "[128,128] K=8 @10"),
        ],
    )
    behind = {
        s: [cell(s, n, "D5") - cell(s, n, "D4") for n in ("10.0", "0.0")]
        for s in ("mit-bih", "eeg", "sine")
    }
    other = behind["eeg"] + behind["sine"]
    a.claim(
        "falls 0.10-0.13 dB behind on MIT-BIH and 0.3-0.7 dB on EEG and the sine",
        f2(min(behind["mit-bih"])),
        f2(max(behind["mit-bih"])),
        f"{min(other):.1f}",
        f"{max(other):.1f}",
    )

    # ------------------------------------------------------------ devices
    pa1 = [r for r in pa_tests if r["test"].startswith("PA1")]
    pa2 = [r for r in pa_tests if r["test"].startswith("PA2")]
    pa3 = next(r for r in pa_tests if r["test"].startswith("PA3"))
    pa4 = [r for r in pa_tests if r["test"].startswith("PA4")]
    a.check("1-bit equivalent at all nine", all(r["b_min"] == "1" for r in pa1))
    a.claim(
        "costing at most 0.22 dB, and 2 bits cost at most 0.05 dB",
        f"{abs(worst['1 bit']):.2f}",
        f"{abs(worst['2 bit']):.2f}",
    )
    sig = [float(r["sigma_max"]) for r in pa2]
    a.check(
        "sigma equivalent to 1.0 at all nine and 1.5 at five",
        min(sig) == 1.0 and sum(s == 1.5 for s in sig) == 5,
    )
    a.claim("equivalent up to \\sigma = 1.0 at all nine points and up to 1.5 at five")
    a.claim(
        "cost at most 0.15 dB at 25%, 0.42 dB at 50% and 0.69 dB at 75%",
        f"{abs(worst['stuck 0.25']):.2f}",
        f"{abs(worst['stuck 0.5']):.2f}",
        f"{abs(worst['stuck 0.75']):.2f}",
    )
    failing = [r for r in perturbed if r["equivalent_to_exact"] != "True"]
    a.claim(
        "Only 12 of the 99 perturbed points fail the equivalence test, seven of them "
        "at 75% stuck",
        str(len(failing)),
        str(len(perturbed)),
    )
    a.check(
        "seven failures at 75% stuck",
        sum(r["level"] == "stuck 0.75" for r in failing) == 7,
    )
    a.claim(
        "Spearman \\rho_s = 0.81 over the 99 points",
        f2(float(pa_post["all"]["spearman"])),
    )
    a.claim("equivalent at all 18", str(len(pa4)))
    a.check("PA4 all equivalent", all(r["equivalent"] == "True" for r in pa4))
    a.claim("measured losses are 0.44 times the predicted", f2(slope))
    a.claim(
        "On the 22 points predicted to lose at least 0.5 dB",
        str(int(float(pa3["big_loss_points"]))),
    )
    a.claim(
        "median absolute error of 0.37 dB against 0.46 dB",
        f2(float(pa3["big_null_median_abs_error_db"])),
        f2(float(pa3["big_model_median_abs_error_db"])),
    )

    def change(n_in: int, hz: str, level: str, key: str = "change_db") -> float:
        return float(
            next(
                r
                for r in pa_cells
                if r["n_in"] == str(n_in)
                and r["encoder_hz"] == hz
                and r["level"] == level
            )[key]
        )

    a.claim(
        "0.49 dB at 2048 channels against 0.29 dB at 128 with 75% stuck and a 40 Hz",
        f"{abs(change(2048, '40.0', 'stuck 0.75')):.2f}",
        f"{abs(change(128, '40.0', 'stuck 0.75')):.2f}",
    )
    a.claim(
        "at 5.19 against 4.84 dB",
        f2(change(2048, "40.0", "stuck 0.75", "ceiling_db")),
        f2(change(128, "40.0", "stuck 0.75", "ceiling_db")),
    )

    # ------------------------------------------------------------ discussion
    a.claim("A second costs 2.5 times the synaptic work and 0.9 dB")
    a.claim(
        "Budget one bit if 0.22 dB is affordable, and expect a quarter of open "
        "devices to cost under 0.2 dB"
    )
    a.check("quarter of open devices under 0.2 dB", abs(worst["stuck 0.25"]) < 0.2)
    a.claim("reference architecture for 1.6 times its synaptic work")
    a.check(
        "eight times the synaptic work (D1 over D2)",
        round(
            cell("mit-bih", "inf", "D1", "synops_per_s")
            / cell("mit-bih", "inf", "D2", "synops_per_s")
        )
        == 8,
    )
    mem = cell("mit-bih", "inf", "D1", "weight_memory_bits") / cell(
        "mit-bih", "inf", "D3", "weight_memory_bits"
    )
    a.claim("30 times the weight memory, addresses included", f"{mem:.0f}")
    encoder_events = cell("mit-bih", "inf", "D2", "spikes_per_s") - 128 * 20
    a.check(
        "about 20 thousand encoder events for D2",
        19_000 < encoder_events < 21_500,
        f"{encoder_events:.0f}",
    )
    a.check("9-bit addresses into 512", math.ceil(math.log2(512)) == 9)
    a.claim("about 20 thousand 9-bit events for D2")
    a.claim("1,024 entries for D2", f"{cell('mit-bih', 'inf', 'D2', 'synapses'):,.0f}")
    a.claim(
        "would hold 64 times as many weights",
        f"{512 * 128 / cell('mit-bih', 'inf', 'D2', 'synapses'):.0f}",
    )


def _rows(path: Path) -> list[Row]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tables_match() -> list[str]:
    committed = ROOT / "paper" / "mcsoc2027" / "figures"
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        mcsoc_figures.FIGURES = Path(tmp)
        mcsoc_figures.ANALYSIS = ANALYSIS
        builders: list[Callable[[], None]] = [
            mcsoc_figures.table_levers,
            mcsoc_figures.table_designs,
            mcsoc_figures.table_devices,
        ]
        for build in builders:
            build()
        for table in Path(tmp).glob("*.tex"):
            if table.read_text() != (committed / table.name).read_text():
                problems.append(f"TABLE DIFFERS from committed: {table.name}")
    return problems


def coverage(audit: Audit, tex: str) -> list[str]:
    body = tex[tex.index("\\begin{abstract}") : tex.index("\\bibliographystyle")]
    body = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", body, flags=re.S
    )
    uncovered = []
    for sentence in re.split(r"(?<=[.;:])\s", normalise(body, source=True)):
        if not re.search(r"\d\.\d|\d{2,}", sentence):
            continue
        if any(chunk in sentence or sentence in chunk for chunk in audit.covered):
            continue
        uncovered.append(sentence.strip())
    return uncovered


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    audit = Audit(tex)
    run(audit)
    audit.failures += tables_match()
    print(f"Number audit: {audit.passed} passed, {len(audit.failures)} failed")
    for failure in audit.failures:
        print("  " + failure)
    rest = coverage(audit, tex)
    print(f"\nSentences with numbers no claim covers ({len(rest)}), for a human check:")
    for sentence in rest:
        print("  - " + sentence[:200])
    sys.exit(1 if audit.failures else 0)


if __name__ == "__main__":
    main()
