"""
Post hoc, for the ICEE 2026 camera-ready: what sets the height of the ceiling.

Reviewer 1 asked why the ceiling differs across signals (1.76-7.48 dB) and for
the effects of intrinsic spike noise, causal readout and temporal filtering to
be separated. This splits the noiseless-input ceiling of R3's rate-controlled,
untrained `[32,16,8]` network into those parts, from the committed C1 grid and
the clean signals alone. Nothing is simulated.

Stage 0 (the Poisson encoder, read out by the 20 ms exponential filter) has a
closed form. Its readout is the expected filtered rate `m` plus spike noise `n`
independent of it, so its affine-corrected S/N against the clean signal `x` is
exactly

    1 / R_0 = 1 / F + 1 / (rho^2 * Lambda)

where `F = rho^2 / (1 - rho^2)` is the S/N of the filter alone (`rho` the
correlation of `x` with `m`) and `Lambda = var(m) / var(n)` is the shot-noise
S/N of `N` Bernoulli encoders, `var(n) = mean_t sum_k d^(2k) p(1 - p) / N`.
Both come from the clean signal and the run's own encoder ceiling, so the
prediction of the measured stage-0 SNR has no free parameter: it tests the
additive-independent-noise assumption of the cascade law at the one stage
where the noise is known exactly. The two LIF stages have no closed form;
their intrinsic S/N is backed out from the measured stage SNRs by the law.

Usage:
    uv run python scripts/paper_icee2026/icee_decomposition.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from neuroflux_core.metrics.cascade import affine_to_excess_db, excess_to_affine_db
from neuroflux_core.metrics.reconstruction import affine_corrected_snr
from scipy.signal import lfilter  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_characterisation"))
import analyse_r3 as r3  # noqa: E402
from r3_common import (  # noqa: E402
    DT,
    DURATION_S,
    best_lag_snr,
    load_signal,
    readout,
    split_floats,
)

OUT = Path("results/icee2026_revision")
CACHE_DIR = "data/mit_bih"
SIGNALS = ("mit-bih", "synthetic-ecg", "eeg", "sine")
DECODER_TAU_MS = 20.0

UNIT_FIELDS = (
    "signal",
    "unit",
    "n_runs",
    "mean_u",
    "cv2_u",
    "filter_causal_db",
    "filter_compensated_db",
    "shot_noise_db",
    "stage0_predicted_db",
    "stage0_measured_db",
    "stage1_measured_db",
    "ceiling_db",
    "lif1_intrinsic_db",
    "lif2_intrinsic_db",
    "share_filter",
    "share_delay",
    "share_encoder_spikes",
    "share_lif1",
    "share_lif2",
)


def inverse_sn(affine_db: float) -> float:
    """`1 / (S/N)` of an affine-corrected SNR in dB; inf when it carries nothing."""
    excess = affine_to_excess_db(affine_db)
    return math.inf if excess == -math.inf else 10.0 ** (-excess / 10.0)


def affine_db_from_inverse(inverse: float) -> float:
    """The affine-corrected SNR in dB that a total `1 / (S/N)` corresponds to."""
    return excess_to_affine_db(-10.0 * math.log10(inverse))


def intrinsic_db(inverse: float) -> float:
    """
    A stage's intrinsic S/N in dB from the inverse S/N it adds.

    `inf` when the stage added nothing or scored above its input, as
    `metrics.cascade.intrinsic_snr_db` does. On EEG the first LIF layer's
    readout scores above the encoder's in every run: under one fixed causal
    readout a stage can recover part of the readout's phase lag, so per-stage
    shares are not meaningful there and the paper does not quote them.
    """
    return math.inf if inverse <= 0.0 else -10.0 * math.log10(inverse)


def encoder_terms(
    clean: np.ndarray, encoder_max_hz: float, n_channels: int
) -> tuple[float, float, float, float, float]:
    """
    Closed-form stage-0 quantities for one noiseless run.

    Returns `(mean_u, cv2_u, inverse F, inverse filter-compensated F,
    inverse rho^2 Lambda)` with `u` the clean signal on the encoder's range.
    """
    u = (clean - clean.min()) / (clean.max() - clean.min())
    p = np.clip(u * encoder_max_hz * DT / 1000.0, 0.0, 1.0)
    decay = math.exp(-DT / DECODER_TAU_MS)
    expected = lfilter([1.0], [1.0, -decay], p)
    variance = lfilter([1.0], [1.0, -decay * decay], p * (1.0 - p))

    filter_db = affine_corrected_snr(clean, expected)
    compensated_db, _ = best_lag_snr(clean, expected)
    rho2 = 1.0 - 10.0 ** (-filter_db / 10.0)
    shot = float(np.var(expected)) / (float(np.mean(variance)) / n_channels)
    return (
        float(np.mean(u)),
        float(np.var(u) / np.mean(u) ** 2),
        inverse_sn(filter_db),
        inverse_sn(compensated_db),
        1.0 / (rho2 * shot),
    )


def unit_rows(signal: str, c1_rows: list[r3.Row]) -> list[dict[str, object]]:
    runs = r3.select(
        c1_rows,
        signal=signal,
        operating_point="controlled",
        epochs="0",
        interference="gaussian",
        input_snr_nominal="inf",
        decoder_tau="20.0",
        verdict="ok",
    )
    by_unit: defaultdict[str, list[r3.Row]] = defaultdict(list)
    for row in runs:
        by_unit[r3.unit_id(row)].append(row)

    out: list[dict[str, object]] = []
    for unit, group in sorted(by_unit.items()):
        values: defaultdict[str, list[float]] = defaultdict(list)
        for row in group:
            clean, _ = load_signal(
                signal, row["unit"], int(row["seed"]), DURATION_S, CACHE_DIR
            )
            width = int(row["layers"].split("-")[0])
            mean_u, cv2, inv_f, inv_f_comp, inv_shot = encoder_terms(
                clean, float(row["encoder_max_hz"]), width
            )
            stages = split_floats(row["stage_affine_db"])
            inv = [inverse_sn(s) for s in stages]
            inv_total = inv[-1]
            inv_lif = [inv[k] - inv[k - 1] for k in range(1, len(inv))]

            values["mean_u"].append(mean_u)
            values["cv2_u"].append(cv2)
            values["filter_causal_db"].append(affine_db_from_inverse(inv_f))
            values["filter_compensated_db"].append(affine_db_from_inverse(inv_f_comp))
            values["shot_noise_db"].append(affine_db_from_inverse(inv_shot))
            values["stage0_predicted_db"].append(
                affine_db_from_inverse(inv_f + inv_shot)
            )
            values["stage0_measured_db"].append(stages[0])
            values["stage1_measured_db"].append(stages[1])
            values["ceiling_db"].append(stages[-1])
            values["lif1_intrinsic_db"].append(intrinsic_db(inv_lif[0]))
            values["lif2_intrinsic_db"].append(intrinsic_db(inv_lif[1]))
            # Shares of the ceiling's inverse S/N. The measured stage 0 is
            # split between filter and spikes in the proportion the closed
            # form gives, so the four shares sum to one by construction.
            inv_spikes_measured = inv[0] - inv_f
            values["share_filter"].append(inv_f / inv_total)
            values["share_delay"].append((inv_f - inv_f_comp) / inv_total)
            values["share_encoder_spikes"].append(inv_spikes_measured / inv_total)
            values["share_lif1"].append(inv_lif[0] / inv_total)
            values["share_lif2"].append(inv_lif[1] / inv_total)

        record: dict[str, object] = {"signal": signal, "unit": unit}
        record["n_runs"] = len(group)
        for name in UNIT_FIELDS[3:]:
            record[name] = float(np.mean(values[name]))
        out.append(record)
    return out


def write(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )
    print(f"  wrote {path}")


CLIP_FIELDS = ("signal", "n_runs", "measured_db", "with_clip_db", "without_clip_db")


def clip_check(c1_rows: list[r3.Row]) -> list[dict[str, object]]:
    """
    Re-simulate the encoder stage with and without the readout's clip.

    The closed form ignores that `readout` clips each trace at the encoder's
    maximum rate. Drawing the 32 Bernoulli encoders afresh and reading them out
    both ways shows whether that clip, rather than the noise model, accounts
    for the gap between prediction and measurement.
    """
    out: list[dict[str, object]] = []
    decay = math.exp(-DT / DECODER_TAU_MS)
    for signal in SIGNALS:
        runs = r3.select(
            c1_rows,
            signal=signal,
            operating_point="controlled",
            epochs="0",
            interference="gaussian",
            input_snr_nominal="inf",
            decoder_tau="20.0",
            verdict="ok",
        )
        measured, clipped, unclipped = [], [], []
        for row in runs:
            seed = int(row["seed"])
            clean, _ = load_signal(signal, row["unit"], seed, DURATION_S, CACHE_DIR)
            width = int(row["layers"].split("-")[0])
            max_hz = float(row["encoder_max_hz"])
            u = (clean - clean.min()) / (clean.max() - clean.min())
            p = np.clip(u * max_hz * DT / 1000.0, 0.0, 1.0)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 60_000]))
            spikes = rng.random((clean.size, width)) < p[:, None]
            clipped.append(
                affine_corrected_snr(
                    clean, readout(spikes, 0.0, 1.0, DECODER_TAU_MS, max_hz)
                )
            )
            traces = lfilter([1.0], [1.0, -decay], spikes.astype(np.float64), axis=0)
            unclipped.append(affine_corrected_snr(clean, traces.mean(axis=1)))
            measured.append(split_floats(row["stage_affine_db"])[0])
        out.append(
            {
                "signal": signal,
                "n_runs": len(runs),
                "measured_db": r3.median(measured),
                "with_clip_db": r3.median(clipped),
                "without_clip_db": r3.median(unclipped),
            }
        )
    return out


RATE_FIELDS = ("unit", "n_seeds", "ceiling_40hz_db", "ceiling_5hz_db", "gain_5hz_db")


def rate_by_record() -> list[dict[str, object]]:
    """
    Per-record ceiling at a 5 Hz against a 40 Hz output, `[32,16,64]`.

    Table I's output-rate row is a median over records with a wide interval;
    R3 section 3.2 found the gain bimodal across records. This is that split
    at the width the table quotes. Descriptive, not a tested hypothesis.
    """
    rows = r3.select(
        r3.load(r3.raw("c3_rate.csv")),
        signal="mit-bih",
        input_snr_nominal="inf",
        layers="32-16-64",
        decoder_tau="20.0",
        verdict="ok",
    )
    ceilings: defaultdict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        output_hz = split_floats(row["layer_rate_targets_hz"])[-1]
        ceilings[(row["unit"], output_hz)].append(float(row["snr_affine_db"]))
    out: list[dict[str, object]] = []
    for unit in sorted({u for u, _ in ceilings}):
        dense = float(np.mean(ceilings[(unit, 40.0)]))
        sparse = float(np.mean(ceilings[(unit, 5.0)]))
        out.append(
            {
                "unit": unit,
                "n_seeds": len(ceilings[(unit, 5.0)]),
                "ceiling_40hz_db": dense,
                "ceiling_5hz_db": sparse,
                "gain_5hz_db": sparse - dense,
            }
        )
    return out


def main() -> None:
    c1_rows = r3.load(r3.raw("c1_flatness.csv"))
    units: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for signal in SIGNALS:
        rows = unit_rows(signal, c1_rows)
        units += rows
        medians: dict[str, object] = {"signal": signal, "unit": "median"}
        medians["n_runs"] = sum(int(str(r["n_runs"])) for r in rows)
        for name in UNIT_FIELDS[3:]:
            medians[name] = r3.median(float(str(r[name])) for r in rows)
        errors = [
            float(str(r["stage0_predicted_db"])) - float(str(r["stage0_measured_db"]))
            for r in rows
        ]
        medians["stage0_rms_error_db"] = float(np.sqrt(np.mean(np.square(errors))))
        summary.append(medians)
        print(
            f"{signal:>14}: ceiling {medians['ceiling_db']:.2f} dB | filter alone "
            f"{medians['filter_causal_db']:.2f} (compensated "
            f"{medians['filter_compensated_db']:.2f}) | stage 0 predicted "
            f"{medians['stage0_predicted_db']:.2f} vs measured "
            f"{medians['stage0_measured_db']:.2f} "
            f"(RMS {medians['stage0_rms_error_db']:.2f})"
        )
    write(OUT / "decomposition_units.csv", UNIT_FIELDS, units)
    write(
        OUT / "decomposition_summary.csv",
        (*UNIT_FIELDS, "stage0_rms_error_db"),
        summary,
    )

    clip = clip_check(c1_rows)
    for row in clip:
        print(
            f"{row['signal']:>14}: encoder stage measured {row['measured_db']:.2f} dB"
            f" | re-simulated with clip {row['with_clip_db']:.2f}"
            f" | without {row['without_clip_db']:.2f}"
        )
    write(OUT / "clip_check.csv", CLIP_FIELDS, clip)

    rates = rate_by_record()
    gains = sorted(float(str(r["gain_5hz_db"])) for r in rates)
    print(f"5 Hz minus 40 Hz per record ([32,16,64]): {[round(g, 2) for g in gains]}")
    write(OUT / "rate_by_record.csv", RATE_FIELDS, rates)


if __name__ == "__main__":
    main()
