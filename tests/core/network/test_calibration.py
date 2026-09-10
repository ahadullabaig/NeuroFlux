"""Tests for per-layer synaptic gain and rate calibration.

Covers the `layer_gains` config field, `FeedForwardNetwork.set_layer_gains`,
and the `calibrate_gain` / `measure_layer_rates` pair. See
`docs/paper_roadmap.md` R4/M1b — every ceiling claim in the paper depends on
these being correct, so the degeneracy guards are tested explicitly.
"""

import numpy as np
import pytest
from neuroflux_core import (
    LIFConfig,
    Network,
    NetworkConfig,
    calibrate_gain,
    measure_layer_rates,
)
from pydantic import ValidationError


def _input_spikes(n_steps: int = 300, n_input: int = 32, p: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n_steps, n_input)) < p


# --------------------------------------------------------------------------
# NetworkConfig.layer_gains
# --------------------------------------------------------------------------


def test_layer_gains_defaults_to_global_gain() -> None:
    """Backward compatibility: no layer_gains means neuron.g_syn everywhere."""
    config = NetworkConfig(layers=[32, 16, 8], neuron=LIFConfig(g_syn=25.0))
    assert config.layer_gains is None
    assert config.gain_for_layer(0) == 25.0
    assert config.gain_for_layer(1) == 25.0


def test_layer_gains_resolves_per_layer() -> None:
    config = NetworkConfig(layers=[32, 16, 8], layer_gains=[10.0, 50.0])
    assert config.gain_for_layer(0) == 10.0
    assert config.gain_for_layer(1) == 50.0


def test_layer_gains_wrong_length_rejected() -> None:
    """3 layers means 2 neuron groups, so 3 gains is a config error."""
    with pytest.raises(ValidationError, match="3 entries but 2 neuron groups"):
        NetworkConfig(layers=[32, 16, 8], layer_gains=[1.0, 2.0, 3.0])


def test_layer_gains_non_positive_rejected() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        NetworkConfig(layers=[32, 16], layer_gains=[0.0])


# --------------------------------------------------------------------------
# FeedForwardNetwork per-layer gain plumbing
# --------------------------------------------------------------------------


def test_network_applies_per_layer_gains_at_construction() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, layer_gains=[7.0, 21.0])
    assert net.neurons[0].config.g_syn == 7.0
    assert net.neurons[1].config.g_syn == 21.0
    assert net.layer_gains == [7.0, 21.0]


def test_default_construction_unchanged() -> None:
    """The default path must still put g_syn=30.0 on every layer."""
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    assert net.layer_gains == [30.0, 30.0]


def test_set_layer_gains_updates_neurons_and_config() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    net.set_layer_gains([2.0, 3.0])
    assert net.layer_gains == [2.0, 3.0]
    assert net.config.layer_gains == [2.0, 3.0]
    assert net.neurons[0].config.g_syn == 2.0


def test_set_layer_gains_preserves_weights() -> None:
    """Calibration must not disturb what the network has learned."""
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    before = [s.weights.copy() for s in net.synapses]
    net.set_layer_gains([5.0, 5.0])
    for w_before, synapse in zip(before, net.synapses, strict=True):
        np.testing.assert_array_equal(w_before, synapse.weights)


def test_set_layer_gains_rejects_bad_input() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    with pytest.raises(ValueError, match="Expected 2 gains, got 1"):
        net.set_layer_gains([1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        net.set_layer_gains([1.0, -1.0])


# --------------------------------------------------------------------------
# measure_layer_rates
# --------------------------------------------------------------------------


def test_measure_layer_rates_returns_one_rate_per_group() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    rates = measure_layer_rates(net, _input_spikes(), dt=1.0)
    assert len(rates) == 2
    assert all(r >= 0.0 for r in rates)


def test_measure_layer_rates_is_deterministic() -> None:
    """Plasticity is disabled, so repeated measurement must agree exactly."""
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    spikes = _input_spikes()
    assert measure_layer_rates(net, spikes, 1.0) == measure_layer_rates(
        net, spikes, 1.0
    )


def test_measure_layer_rates_rejects_wrong_shape() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    with pytest.raises(ValueError, match="must be 2-D"):
        measure_layer_rates(net, np.zeros(32, dtype=bool), dt=1.0)
    with pytest.raises(ValueError, match="width 8 but the input layer is 32"):
        measure_layer_rates(net, np.zeros((10, 8), dtype=bool), dt=1.0)


def test_measure_layer_rates_does_not_train() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    before = [s.weights.copy() for s in net.synapses]
    measure_layer_rates(net, _input_spikes(), dt=1.0)
    for w_before, synapse in zip(before, net.synapses, strict=True):
        np.testing.assert_array_equal(w_before, synapse.weights)


# --------------------------------------------------------------------------
# calibrate_gain
# --------------------------------------------------------------------------


def test_calibrate_gain_drives_all_layers_to_target() -> None:
    """The core contract: every layer lands near the target rate."""
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    spikes = _input_spikes()
    calibrate_gain(net, spikes, dt=1.0, target_rate_hz=40.0, tol_hz=0.5)
    rates = measure_layer_rates(net, spikes, dt=1.0)
    for rate in rates:
        assert abs(rate - 40.0) < 3.0, f"layer landed at {rate:.1f} Hz, not ~40 Hz"


def test_calibrate_gain_fixes_silent_deep_layer() -> None:
    """
    Regression guard for the degeneracy that motivated this module.

    At the default g_syn=30.0 the second group is completely silent under this
    input; calibration must revive it. See docs/paper_roadmap.md Finding 3.
    """
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    spikes = _input_spikes()

    before = measure_layer_rates(net, spikes, dt=1.0)
    assert before[1] == 0.0, "expected the default gain to silence layer 2"

    calibrate_gain(net, spikes, dt=1.0, target_rate_hz=40.0)
    after = measure_layer_rates(net, spikes, dt=1.0)
    assert after[1] > 30.0, f"layer 2 still near-silent at {after[1]:.1f} Hz"


def test_calibrate_gain_applies_in_place_and_returns_same_gains() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    gains = calibrate_gain(net, _input_spikes(), dt=1.0, target_rate_hz=40.0)
    assert gains == net.layer_gains


def test_calibrate_gain_preserves_weights() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    before = [s.weights.copy() for s in net.synapses]
    calibrate_gain(net, _input_spikes(), dt=1.0, target_rate_hz=40.0)
    for w_before, synapse in zip(before, net.synapses, strict=True):
        np.testing.assert_array_equal(w_before, synapse.weights)


def test_calibrate_gain_rejects_bad_arguments() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    spikes = _input_spikes()
    with pytest.raises(ValueError, match="target_rate_hz must be positive"):
        calibrate_gain(net, spikes, dt=1.0, target_rate_hz=0.0)
    with pytest.raises(ValueError, match="0 < low < high"):
        calibrate_gain(net, spikes, dt=1.0, bounds=(10.0, 1.0))


@pytest.mark.parametrize("n_input", [8, 32, 128])
def test_calibrate_gain_works_across_input_widths(n_input: int) -> None:
    """
    The whole point: a fixed gain cannot serve every width, a calibrated one can.

    This is what makes the M1a capacity sweep a controlled comparison.
    """
    net = Network.feedforward(layers=[n_input, 16, 8], seed=42, init_scheme="scaled")
    spikes = _input_spikes(n_input=n_input)
    calibrate_gain(net, spikes, dt=1.0, target_rate_hz=40.0)
    rates = measure_layer_rates(net, spikes, dt=1.0)
    for rate in rates:
        assert abs(rate - 40.0) < 5.0, f"n_input={n_input} landed at {rate:.1f} Hz"
