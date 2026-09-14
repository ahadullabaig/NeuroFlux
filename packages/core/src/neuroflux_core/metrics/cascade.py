"""
The inverse-SNR cascade law for staged spiking pipelines.

Every stage of a rate-coded pipeline does two things to the signal it
receives: it applies some transfer gain, and it adds spiking noise of its
own. Write stage `k`'s population readout as

    y_k = G_k * y_{k-1} + eta_k

where `eta_k` is the stage's own spike-sampling noise, independent of
everything upstream. Signal power scales as `G_k^2 S_{k-1}` and noise power
as `G_k^2 N_{k-1} + var(eta_k)`, so the gain cancels out of the ratio and

    1/SNR_k = 1/SNR_{k-1} + var(eta_k) / (G_k^2 S_{k-1})
            = 1/SNR_{k-1} + 1/Lambda_k

with `Lambda_k` the stage's **intrinsic SNR** — what it would deliver from a
noiseless input. Telescoping over the pipeline:

    1/SNR_out = 1/SNR_in + sum_k 1/Lambda_k

Inverse SNRs add. This is Friis's noise-figure formula for a cascade of
amplifiers, and it has the same structure as the Fisher-information
saturation of Moreno-Bote et al. (2014), `I_N = I_0 / (1 + eps * I_0)`.

Three consequences, all of which the repository had already measured before
the law was written down:

1. **There is a ceiling, and it is the clean-signal output.** As
   `SNR_in -> inf` the first term vanishes and `SNR_out` approaches
   `1 / sum_k 1/Lambda_k`, a finite constant set entirely by the pipeline.
   Feeding the network a *noiseless* signal measures that constant directly.
2. **Depth can only hurt.** Every stage contributes a strictly positive
   `1/Lambda_k`, so appending a layer always lowers `SNR_out`.
3. **Only the weakest stages matter.** Improving a stage whose `1/Lambda_k`
   is a small part of the sum moves the total almost not at all, which is why
   widening the output population or retuning the decoder buys fractions of
   a dB while removing a layer buys whole dB.

The law is stated in linear power ratios; every function here takes and
returns dB for consistency with the rest of `metrics`.

**Which SNR the law is about.** The derivation is in terms of signal-to-noise
power, `S/N`. The repository scores reconstructions with *affine-corrected*
SNR (`metrics.reconstruction`), and that is not the same quantity: a best-fit
scale and offset leave a residual of `var(x) * (1 - rho^2)`, so the affine SNR
is `1 / (1 - rho^2)`, and for a signal plus independent noise that is exactly
`1 + S/N`. It can never fall below 0 dB. Feeding affine values straight into
the law therefore breaks it wherever `S/N` is small — on MIT-BIH it
over-predicts the loss by about 2 dB at −20 dB input (`docs/r3_findings.md`
C1). Convert with `affine_to_excess_db` before applying the law, and back with
`excess_to_affine_db`; the two forms agree once every term is well above 0 dB.

See `docs/m3_findings.md` for the derivation of `Lambda_k` itself from spike
statistics, and for the measurements that test all of the above.
"""

import math

# Below this linear SNR a stage carries essentially no signal, and inverting
# the ratio amplifies float noise into meaningless numbers.
_MIN_LINEAR_SNR = 1e-12


def db_to_linear(snr_db: float) -> float:
    """Convert an SNR in dB to a linear power ratio."""
    return float(10.0 ** (snr_db / 10.0))


def linear_to_db(snr_linear: float) -> float:
    """
    Convert a linear power ratio to dB.

    Returns:
        `-inf` for a ratio of zero and `inf` for an infinite ratio, so that a
        lossless or a dead stage both propagate rather than raising.
    """
    if snr_linear <= 0.0:
        return -math.inf
    if math.isinf(snr_linear):
        return math.inf
    return float(10.0 * math.log10(snr_linear))


def affine_to_excess_db(snr_affine_db: float) -> float:
    """
    The signal-to-noise ratio an affine-corrected SNR implies, in dB.

    Affine-corrected SNR is `1 + S/N` in linear terms (see the module
    docstring), so this returns `10 * log10(10^(x / 10) - 1)`.

    Args:
        snr_affine_db: An affine-corrected SNR in dB.

    Returns:
        `S/N` in dB: `inf` for an infinite input, `-inf` at or below 0 dB, where
        the reconstruction carries no linear information about the signal.
    """
    if math.isinf(snr_affine_db):
        return snr_affine_db
    return linear_to_db(db_to_linear(snr_affine_db) - 1.0)


def excess_to_affine_db(snr_excess_db: float) -> float:
    """
    The affine-corrected SNR a given `S/N` produces, in dB.

    The inverse of `affine_to_excess_db`: `10 * log10(1 + 10^(x / 10))`.

    Args:
        snr_excess_db: A signal-to-noise ratio in dB; `-inf` means no signal.

    Returns:
        The affine-corrected SNR in dB, never below 0 dB.
    """
    if math.isinf(snr_excess_db):
        return 0.0 if snr_excess_db < 0.0 else math.inf
    return linear_to_db(1.0 + db_to_linear(snr_excess_db))


def intrinsic_snr_db(input_snr_db: float, output_snr_db: float) -> float:
    """
    Back out a stage's intrinsic SNR from what it received and what it produced.

    Inverts `1/SNR_out = 1/SNR_in + 1/Lambda` for `Lambda`, the SNR the stage
    would deliver from a noiseless input. This is the quantity to compare
    across stages: unlike the retention ratio it does not depend on how good
    the upstream signal happened to be.

    Args:
        input_snr_db: SNR entering the stage, in dB.
        output_snr_db: SNR leaving the stage, in dB.

    Returns:
        The intrinsic SNR in dB. `inf` when the stage lost nothing (or
        apparently gained, which measurement noise can produce and which the
        law forbids) — read that as "this stage is not a binding constraint".

    Raises:
        ValueError: If either SNR is not finite.
    """
    if not math.isfinite(input_snr_db) or not math.isfinite(output_snr_db):
        raise ValueError("Both SNRs must be finite to infer an intrinsic SNR.")

    inv_out = 1.0 / max(db_to_linear(output_snr_db), _MIN_LINEAR_SNR)
    inv_in = 1.0 / max(db_to_linear(input_snr_db), _MIN_LINEAR_SNR)
    inv_lambda = inv_out - inv_in

    if inv_lambda <= 0.0:
        return math.inf
    return linear_to_db(1.0 / inv_lambda)


def cascade_snr_db(input_snr_db: float, intrinsic_snr_dbs: list[float]) -> float:
    """
    Predict a pipeline's output SNR from its input SNR and per-stage intrinsics.

    This is the law's forward direction and its falsifiable content: measure
    each `Lambda_k` once (from a clean-signal run), then the output at *any*
    input SNR is fixed with no further free parameters.

    Args:
        input_snr_db: SNR available at the input, in dB.
        intrinsic_snr_dbs: One intrinsic SNR per stage, in dB. Infinite
            entries are lossless stages and contribute nothing.

    Returns:
        Predicted output SNR in dB.

    Raises:
        ValueError: If `input_snr_db` is not finite.
    """
    if not math.isfinite(input_snr_db):
        raise ValueError("input_snr_db must be finite.")

    inv_total = 1.0 / max(db_to_linear(input_snr_db), _MIN_LINEAR_SNR)
    for stage_db in intrinsic_snr_dbs:
        if math.isinf(stage_db) and stage_db > 0.0:
            continue
        inv_total += 1.0 / max(db_to_linear(stage_db), _MIN_LINEAR_SNR)

    return linear_to_db(1.0 / inv_total)


def ceiling_db(intrinsic_snr_dbs: list[float]) -> float:
    """
    The output SNR a pipeline approaches as its input becomes noiseless.

    The `SNR_in -> inf` limit of `cascade_snr_db`. No amount of input quality
    lifts a pipeline past this, which is what makes the measured ceiling a
    property of the architecture rather than of the data.

    Args:
        intrinsic_snr_dbs: One intrinsic SNR per stage, in dB.

    Returns:
        The ceiling in dB. `inf` when every stage is lossless.
    """
    inv_total = 0.0
    for stage_db in intrinsic_snr_dbs:
        if math.isinf(stage_db) and stage_db > 0.0:
            continue
        inv_total += 1.0 / max(db_to_linear(stage_db), _MIN_LINEAR_SNR)

    if inv_total <= 0.0:
        return math.inf
    return linear_to_db(1.0 / inv_total)


def retention(input_snr_db: float, intrinsic_snr_db_value: float) -> float:
    """
    Fraction of the incoming SNR a stage passes on, as a linear ratio.

    `SNR_out / SNR_in = 1 / (1 + SNR_in / Lambda)`. The point of expressing it
    this way is that retention is **not** a constant of the stage: it falls as
    the incoming signal improves, because a fixed amount of added noise costs
    proportionally more when there is less noise already present. A measured
    retention figure is therefore only meaningful alongside the input SNR it
    was measured at.

    Args:
        input_snr_db: SNR entering the stage, in dB.
        intrinsic_snr_db_value: The stage's intrinsic SNR, in dB.

    Returns:
        Retention as a linear ratio in (0, 1].

    Raises:
        ValueError: If `input_snr_db` is not finite.
    """
    if not math.isfinite(input_snr_db):
        raise ValueError("input_snr_db must be finite.")
    if math.isinf(intrinsic_snr_db_value) and intrinsic_snr_db_value > 0.0:
        return 1.0

    ratio = db_to_linear(input_snr_db) / max(
        db_to_linear(intrinsic_snr_db_value), _MIN_LINEAR_SNR
    )
    return float(1.0 / (1.0 + ratio))
