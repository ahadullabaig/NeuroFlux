"""Tests for the inverse-SNR cascade law.

This module is the paper's Section III expressed in code, so the tests are
written as the law's claims rather than as coverage of the functions:

  - inverse SNRs add, and `intrinsic_snr_db` inverts that exactly;
  - depth is monotonically harmful;
  - the ceiling is the noiseless-input limit;
  - retention is not a property of a stage — it depends on what arrives.

The regression test at the bottom pins the measured shipped-configuration
numbers, so a change to LIF dynamics or the encoders that moved the ceiling
would surface here rather than in a re-run of the experiments.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from neuroflux_core.metrics.cascade import (
    affine_to_excess_db,
    cascade_snr_db,
    ceiling_db,
    db_to_linear,
    excess_to_affine_db,
    intrinsic_snr_db,
    linear_to_db,
    retention,
)

# Measured in `scripts/m3_bound/run_m3.py` on the shipped [32,16,8] tau=20 ms
# configuration with a noiseless input: encoder, then two LIF layers.
SHIPPED_LAMBDAS = [3.87, 8.59, 7.76]
SHIPPED_CEILING = 1.45
# Same measurement on the wide [128,128,128] tau=4 ms configuration, where
# the weak stage is the first LIF layer rather than the encoder.
WIDE_LAMBDAS = [9.12, 3.13, 5.72]
WIDE_CEILING = 0.57


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------


def test_db_linear_round_trip() -> None:
    for db in [-20.0, -3.0, 0.0, 1.45, 10.0, 40.0]:
        assert linear_to_db(db_to_linear(db)) == pytest.approx(db, abs=1e-9)


def test_zero_db_is_unit_ratio() -> None:
    assert db_to_linear(0.0) == pytest.approx(1.0)


def test_linear_to_db_handles_degenerate_ratios() -> None:
    assert linear_to_db(0.0) == -math.inf
    assert linear_to_db(-1.0) == -math.inf
    assert linear_to_db(math.inf) == math.inf


# --------------------------------------------------------------------------
# the law itself
# --------------------------------------------------------------------------


def test_inverse_snrs_add() -> None:
    """The defining identity, checked in linear power ratios."""
    result = cascade_snr_db(20.0, [10.0, 6.0])
    expected = 1.0 / (1.0 / db_to_linear(20.0) + 1.0 / 10.0 + 1.0 / db_to_linear(6.0))
    assert db_to_linear(result) == pytest.approx(expected, rel=1e-9)


def test_intrinsic_snr_inverts_cascade() -> None:
    """`intrinsic_snr_db` must recover exactly the stage it was given."""
    for input_db, lam in [(20.0, 8.0), (0.0, 3.0), (-5.0, 12.0), (40.0, 1.0)]:
        output_db = cascade_snr_db(input_db, [lam])
        assert intrinsic_snr_db(input_db, output_db) == pytest.approx(lam, abs=1e-6)


def test_single_stage_matches_multi_stage_decomposition() -> None:
    """Two stages applied together equal the same two applied in sequence."""
    together = cascade_snr_db(15.0, [9.0, 4.0])
    sequential = cascade_snr_db(cascade_snr_db(15.0, [9.0]), [4.0])
    assert together == pytest.approx(sequential, abs=1e-9)


def test_no_stages_is_lossless() -> None:
    assert cascade_snr_db(12.5, []) == pytest.approx(12.5)


def test_lossless_stage_contributes_nothing() -> None:
    assert cascade_snr_db(12.5, [math.inf, math.inf]) == pytest.approx(12.5)


def test_intrinsic_snr_of_lossless_stage_is_infinite() -> None:
    assert intrinsic_snr_db(10.0, 10.0) == math.inf


def test_intrinsic_snr_of_apparent_gain_is_infinite() -> None:
    """
    The law forbids a stage improving SNR, but measurement noise produces it.

    Reporting `inf` says "not a binding constraint" rather than inventing a
    negative noise power.
    """
    assert intrinsic_snr_db(10.0, 11.0) == math.inf


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------


def test_depth_is_monotonically_harmful() -> None:
    """Every added stage strictly lowers the output. This is why depth hurts."""
    stages = [9.0, 7.0, 12.0, 5.0]
    previous = cascade_snr_db(20.0, [])
    for depth in range(1, len(stages) + 1):
        current = cascade_snr_db(20.0, stages[:depth])
        assert current < previous
        previous = current


def test_output_never_exceeds_input_or_any_stage() -> None:
    stages = [9.0, 3.5, 14.0]
    out = cascade_snr_db(20.0, stages)
    assert out < 20.0
    assert all(out < stage for stage in stages)


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------


def test_ceiling_is_the_noiseless_input_limit() -> None:
    stages = [3.87, 8.59, 7.76]
    assert cascade_snr_db(200.0, stages) == pytest.approx(ceiling_db(stages), abs=1e-6)


def test_ceiling_of_single_stage_is_that_stage() -> None:
    assert ceiling_db([6.5]) == pytest.approx(6.5)


def test_ceiling_of_lossless_pipeline_is_infinite() -> None:
    assert ceiling_db([math.inf]) == math.inf
    assert ceiling_db([]) == math.inf


def test_input_snr_cannot_lift_output_past_the_ceiling() -> None:
    """The central claim: output quality is bounded regardless of input."""
    stages = [3.87, 8.59, 7.76]
    limit = ceiling_db(stages)
    for input_db in [0.0, 10.0, 40.0, 80.0, 160.0]:
        assert cascade_snr_db(input_db, stages) <= limit + 1e-9


# --------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------


def test_retention_matches_the_cascade() -> None:
    for input_db, lam in [(8.56, 2.96), (1.90, 5.95), (20.0, 10.0)]:
        out_db = cascade_snr_db(input_db, [lam])
        expected = db_to_linear(out_db) / db_to_linear(input_db)
        assert retention(input_db, lam) == pytest.approx(expected, rel=1e-9)


def test_retention_falls_as_input_improves() -> None:
    """
    Retention is not a constant of a stage.

    M2 reported "each LIF layer retains ~22%" as if it were; the same stage
    retains a different fraction depending on what reaches it, which is why
    the intrinsic SNR is the quantity to compare across stages.
    """
    values = [retention(db, 6.0) for db in [-10.0, 0.0, 10.0, 20.0, 30.0]]
    assert all(b < a for a, b in zip(values, values[1:], strict=False))


def test_retention_is_bounded() -> None:
    for input_db in [-20.0, 0.0, 30.0]:
        assert 0.0 < retention(input_db, 6.0) <= 1.0


def test_lossless_stage_retains_everything() -> None:
    assert retention(10.0, math.inf) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_intrinsic_snr_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        intrinsic_snr_db(bad, 1.0)
    with pytest.raises(ValueError, match="finite"):
        intrinsic_snr_db(1.0, bad)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_cascade_and_retention_reject_non_finite_input(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        cascade_snr_db(bad, [5.0])
    with pytest.raises(ValueError, match="finite"):
        retention(bad, 5.0)


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------


@given(
    input_db=st.floats(min_value=-30.0, max_value=60.0),
    stages=st.lists(st.floats(min_value=-10.0, max_value=40.0), min_size=1, max_size=6),
)
def test_output_is_below_input_and_every_stage(
    input_db: float, stages: list[float]
) -> None:
    out = cascade_snr_db(input_db, stages)
    assert out <= input_db + 1e-9
    assert all(out <= stage + 1e-9 for stage in stages)


@given(
    input_db=st.floats(min_value=-30.0, max_value=60.0),
    stages=st.lists(st.floats(min_value=-10.0, max_value=40.0), min_size=1, max_size=6),
)
def test_stage_order_does_not_matter(input_db: float, stages: list[float]) -> None:
    """Inverse SNRs add, and addition commutes — where a loss sits is irrelevant."""
    assert cascade_snr_db(input_db, stages) == pytest.approx(
        cascade_snr_db(input_db, list(reversed(stages))), abs=1e-9
    )


# --------------------------------------------------------------------------
# regression — the measured pipeline
# --------------------------------------------------------------------------


def test_shipped_configuration_ceiling() -> None:
    """Pins the measured ceiling of the shipped [32,16,8] tau=20 ms pipeline."""
    assert ceiling_db(SHIPPED_LAMBDAS) == pytest.approx(SHIPPED_CEILING, abs=0.05)


def test_shipped_configuration_predicts_measured_output() -> None:
    """
    The M3 headline, as a test.

    Intrinsic SNRs measured on a noiseless input predict the 20 dB-input
    output to within the seed-to-seed spread, with nothing fitted. Measured:
    1.33 dB at a 19.47 dB input.
    """
    assert cascade_snr_db(19.47, SHIPPED_LAMBDAS) == pytest.approx(1.33, abs=0.15)


def test_the_binding_stage_depends_on_configuration() -> None:
    """
    M2's "depth dominates" rule is configuration-specific, not a law.

    In the shipped narrow/slow configuration the *encoder* is the weakest
    stage; widening the layers and shortening tau moves the weak point onto
    the first LIF layer. This is M2 §4's "redistribution" finding, and the
    cascade law is what explains it: only the smallest Lambda matters, and
    which stage holds it is not fixed.
    """
    assert SHIPPED_LAMBDAS.index(min(SHIPPED_LAMBDAS)) == 0
    assert WIDE_LAMBDAS.index(min(WIDE_LAMBDAS)) == 1


def test_improving_the_weakest_stage_helps_most() -> None:
    """A fixed dB of improvement is worth most where Lambda is smallest."""
    for lambdas in (SHIPPED_LAMBDAS, WIDE_LAMBDAS):
        base = ceiling_db(lambdas)
        gains = []
        for i in range(len(lambdas)):
            lifted = list(lambdas)
            lifted[i] += 3.0
            gains.append(ceiling_db(lifted) - base)
        assert gains.index(max(gains)) == lambdas.index(min(lambdas))


def test_removing_the_weakest_stage_helps_most() -> None:
    """Which stage to delete is decided by Lambda, not by position."""
    for lambdas in (SHIPPED_LAMBDAS, WIDE_LAMBDAS):
        base = ceiling_db(lambdas)
        gains = [
            ceiling_db(lambdas[:i] + lambdas[i + 1 :]) - base
            for i in range(len(lambdas))
        ]
        assert gains.index(max(gains)) == lambdas.index(min(lambdas))


def test_strong_stages_are_nearly_free_to_ignore() -> None:
    """
    Why Findings 1-5 all read flat.

    Improving every stage *except* the weakest by 3 dB moves the ceiling less
    than improving the weakest alone by the same amount.
    """
    for lambdas in (SHIPPED_LAMBDAS, WIDE_LAMBDAS):
        weakest = lambdas.index(min(lambdas))
        base = ceiling_db(lambdas)
        others = ceiling_db(
            [s if i == weakest else s + 3.0 for i, s in enumerate(lambdas)]
        )
        only_weakest = ceiling_db(
            [s + 3.0 if i == weakest else s for i, s in enumerate(lambdas)]
        )
        assert only_weakest - base > others - base


def test_wide_configuration_predicts_measured_output() -> None:
    """The same zero-parameter prediction on an independent configuration."""
    assert ceiling_db(WIDE_LAMBDAS) == pytest.approx(WIDE_CEILING, abs=0.05)
    assert cascade_snr_db(19.47, WIDE_LAMBDAS) == pytest.approx(0.46, abs=0.15)


def test_np_float_inputs_are_accepted() -> None:
    """Experiment scripts pass numpy scalars straight from `np.mean`."""
    assert cascade_snr_db(
        float(np.float64(19.47)), [float(np.float64(x)) for x in SHIPPED_LAMBDAS]
    ) == pytest.approx(1.33, abs=0.15)


# --------------------------------------------------------------------------
# Affine-corrected SNR is 1 + S/N — the law must be applied to S/N
# --------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [-20.0, -5.0, 0.0, 3.0, 20.0])
def test_affine_snr_of_signal_plus_noise_is_one_plus_s_over_n(snr_db: float) -> None:
    """The identity the conversion rests on, checked on data rather than algebra."""
    from neuroflux_core.metrics.reconstruction import affine_corrected_snr

    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 400 * np.pi, 200_000))
    noise = rng.normal(0.0, 1.0, clean.size)
    noise *= np.sqrt(np.var(clean) / db_to_linear(snr_db) / np.var(noise))
    measured = affine_corrected_snr(clean, 0.3 * (clean + noise) - 2.0)
    assert affine_to_excess_db(measured) == pytest.approx(snr_db, abs=0.05)


def test_excess_and_affine_round_trip() -> None:
    for value in (-30.0, -3.0, 0.0, 1.7, 25.0):
        assert affine_to_excess_db(excess_to_affine_db(value)) == pytest.approx(value)


def test_excess_conversion_limits() -> None:
    assert affine_to_excess_db(0.0) == -math.inf
    assert affine_to_excess_db(math.inf) == math.inf
    assert excess_to_affine_db(-math.inf) == 0.0
    assert excess_to_affine_db(math.inf) == math.inf
    # Well above 0 dB the two scales agree.
    assert affine_to_excess_db(40.0) == pytest.approx(40.0, abs=1e-3)


def test_applying_the_law_to_affine_values_overstates_low_snr_loss() -> None:
    """
    The R3 C1 correction in miniature: a 1.8 dB-ceiling pipeline fed a −20 dB
    input. In S/N terms the output is a hair above 0 dB affine; applied to the
    affine values directly, the law predicts about −2 dB, which the metric
    cannot even produce.
    """
    ceiling, input_db = 1.8, excess_to_affine_db(-20.0)
    direct = cascade_snr_db(input_db, [ceiling])
    corrected = excess_to_affine_db(
        cascade_snr_db(affine_to_excess_db(input_db), [affine_to_excess_db(ceiling)])
    )
    assert direct < -1.5
    assert 0.0 < corrected < 0.05
