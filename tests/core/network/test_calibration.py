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
    PoissonRateEncoder,
    calibrate_encoder_rate,
    calibrate_gain,
    expected_encoder_rate_hz,
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


def test_calibrate_gain_accepts_one_target_per_layer() -> None:
    """An output layer can be held sparse while the layer feeding it stays dense."""
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    spikes = _input_spikes(n_steps=2000)
    calibrate_gain(net, spikes, dt=1.0, target_rate_hz=[40.0, 5.0], tol_hz=0.1)
    rates = measure_layer_rates(net, spikes, dt=1.0)
    assert abs(rates[0] - 40.0) < 5.0, f"layer 1 landed at {rates[0]:.1f} Hz"
    assert abs(rates[1] - 5.0) < 1.0, f"layer 2 landed at {rates[1]:.1f} Hz"


def test_calibrate_gain_rejects_a_target_list_of_the_wrong_length() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42)
    with pytest.raises(ValueError, match="2 neuron groups"):
        calibrate_gain(net, _input_spikes(), dt=1.0, target_rate_hz=[40.0])
    with pytest.raises(ValueError, match="target_rate_hz must be positive"):
        calibrate_gain(net, _input_spikes(), dt=1.0, target_rate_hz=[40.0, -1.0])


def test_calibrating_a_trained_network_keeps_what_it_learned() -> None:
    """
    Regression guard: measuring rates used to call `reset_state`, which restores
    the initial weights, so recalibrating after training discarded the training.
    """
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    spikes = _drive()
    for t in range(spikes.shape[0]):
        net.step(spikes[t], 1.0, training=True)
    trained = [s.weights.copy() for s in net.synapses]
    initial = [s.initial_weights_copy.copy() for s in net.synapses]
    assert any(
        not np.array_equal(a, b) for a, b in zip(initial, trained, strict=True)
    ), "training did not change any weights; the test cannot detect the reset"

    calibrate_gain(net, _input_spikes(), dt=1.0, target_rate_hz=40.0)

    for w_trained, synapse in zip(trained, net.synapses, strict=True):
        np.testing.assert_array_equal(w_trained, synapse.weights)


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


# --------------------------------------------------------------------------
# reset_transient_state — weights must survive an epoch boundary
# --------------------------------------------------------------------------


def _drive(n_steps: int = 200, n_input: int = 32) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.random((n_steps, n_input)) < 0.4


def test_reset_state_discards_learned_weights() -> None:
    """
    Documents the footgun `reset_transient_state` exists to avoid.

    `STDPSynapse.reset_state` restores `initial_weights_copy`, so a training
    loop that calls `network.reset_state()` between epochs silently throws away
    everything it learned.
    """
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")
    initial = [s.weights.copy() for s in net.synapses]

    spikes = _drive()
    for t in range(spikes.shape[0]):
        net.step(spikes[t], 1.0, training=True)

    trained = [s.weights.copy() for s in net.synapses]
    assert any(
        not np.array_equal(a, b) for a, b in zip(initial, trained, strict=True)
    ), "training did not change any weights; the test cannot detect the reset"

    net.reset_state()
    for w_initial, synapse in zip(initial, net.synapses, strict=True):
        np.testing.assert_array_equal(w_initial, synapse.weights)


def test_reset_transient_state_preserves_learned_weights() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")

    spikes = _drive()
    for t in range(spikes.shape[0]):
        net.step(spikes[t], 1.0, training=True)
    trained = [s.weights.copy() for s in net.synapses]

    net.reset_transient_state()
    for w_trained, synapse in zip(trained, net.synapses, strict=True):
        np.testing.assert_array_equal(w_trained, synapse.weights)


def test_reset_transient_state_clears_neuron_and_trace_state() -> None:
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="scaled")

    spikes = _drive()
    for t in range(spikes.shape[0]):
        net.step(spikes[t], 1.0, training=True)

    net.reset_transient_state()
    for neuron_group in net.neurons:
        np.testing.assert_allclose(neuron_group.v, neuron_group.config.v_rest)
        np.testing.assert_allclose(neuron_group.refractory_time, 0.0)
    for synapse in net.synapses:
        np.testing.assert_allclose(synapse.trace_pre, 0.0)
        np.testing.assert_allclose(synapse.trace_post, 0.0)


class TestEncoderRateCalibration:
    """The encoder stage, which `calibrate_gain` cannot reach."""

    DT = 4.0

    @staticmethod
    def _spiky(baseline_fraction: float, n: int = 5000) -> np.ndarray:
        """A signal that sits at a low baseline except for occasional peaks."""
        rng = np.random.default_rng(0)
        signal = rng.uniform(0.0, 0.1, size=n)
        peaks = rng.random(n) > baseline_fraction
        signal[peaks] = rng.uniform(0.5, 1.0, size=int(peaks.sum()))
        return signal

    def test_the_expected_rate_matches_the_encoder_empirically(self) -> None:
        signal = self._spiky(0.8)
        drive = np.tile(signal, (64, 1)).T
        encoder = PoissonRateEncoder(
            v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0, rng=np.random.default_rng(1)
        )
        spikes = np.stack(
            [encoder.encode(drive[t], self.DT) for t in range(len(drive))]
        )
        empirical = float(spikes.mean()) * 1000.0 / self.DT
        expected = expected_encoder_rate_hz(signal, 0.0, 1.0, 100.0, self.DT)
        assert empirical == pytest.approx(expected, rel=0.02)

    def test_the_expected_rate_respects_the_per_step_probability_clip(self) -> None:
        # At dt = 4 ms a channel cannot exceed 250 Hz, however high the ceiling.
        signal = np.ones(100)
        assert expected_encoder_rate_hz(signal, 0.0, 1.0, 10_000.0, self.DT) == (
            pytest.approx(250.0)
        )

    @pytest.mark.parametrize("baseline_fraction", [0.5, 0.8, 0.95])
    def test_calibration_hits_the_target_whatever_the_amplitude_shape(
        self, baseline_fraction: float
    ) -> None:
        signal = self._spiky(baseline_fraction)
        ceiling = calibrate_encoder_rate(signal, 0.0, 1.0, self.DT, target_rate_hz=40.0)
        rate = expected_encoder_rate_hz(signal, 0.0, 1.0, ceiling, self.DT)
        assert rate == pytest.approx(40.0, abs=1e-2)

    def test_a_baseline_heavy_signal_needs_a_higher_ceiling(self) -> None:
        # The confound this exists to remove: the same ceiling gives different
        # rates on signals whose amplitude sits in different places.
        sparse_peaks = calibrate_encoder_rate(self._spiky(0.95), 0.0, 1.0, self.DT)
        dense_peaks = calibrate_encoder_rate(self._spiky(0.5), 0.0, 1.0, self.DT)
        assert sparse_peaks > dense_peaks

    def test_calibration_works_past_the_probability_clip(self) -> None:
        # A target that needs a nominal ceiling above 1000/dt must still land,
        # which a closed-form inverse would miss.
        signal = self._spiky(0.95)
        ceiling = calibrate_encoder_rate(signal, 0.0, 1.0, self.DT, target_rate_hz=40.0)
        assert ceiling > 250.0
        assert expected_encoder_rate_hz(signal, 0.0, 1.0, ceiling, self.DT) == (
            pytest.approx(40.0, abs=1e-2)
        )

    def test_an_unreachable_target_raises(self) -> None:
        with pytest.raises(ValueError, match="unreachable"):
            calibrate_encoder_rate(np.full(100, 0.5), 0.0, 1.0, self.DT, 300.0)

    def test_a_signal_entirely_at_v_min_cannot_fire(self) -> None:
        with pytest.raises(ValueError, match="unreachable"):
            calibrate_encoder_rate(np.zeros(100), 0.0, 1.0, self.DT, 1.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"v_min": 1.0, "v_max": 1.0}, "strictly greater"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"target_rate_hz": 0.0}, "target_rate_hz must be positive"),
        ],
    )
    def test_bad_arguments_raise(self, kwargs: dict[str, float], message: str) -> None:
        arguments: dict[str, float] = {
            "v_min": 0.0,
            "v_max": 1.0,
            "dt": self.DT,
            "target_rate_hz": 40.0,
        }
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            calibrate_encoder_rate(
                np.linspace(0.0, 1.0, 100),
                arguments["v_min"],
                arguments["v_max"],
                arguments["dt"],
                arguments["target_rate_hz"],
            )

    def test_iteration_exhaustion_returns_the_bracket_midpoint(self) -> None:
        signal = self._spiky(0.8)
        ceiling = calibrate_encoder_rate(
            signal, 0.0, 1.0, self.DT, target_rate_hz=40.0, tol_hz=0.0, max_iter=3
        )
        assert ceiling > 0.0
