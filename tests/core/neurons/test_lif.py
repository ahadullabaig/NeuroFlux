import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from neuroflux_core.io.config import LIFConfig
from neuroflux_core.neurons.lif import LIFNeuron


def test_lif_initialization() -> None:
    config = LIFConfig(v_rest=-70.0)
    neurons = LIFNeuron(n_neurons=10, config=config)

    assert neurons.n_neurons == 10
    assert np.allclose(neurons.v, -70.0)
    assert np.allclose(neurons.refractory_time, 0.0)


def test_lif_reset() -> None:
    config = LIFConfig(v_rest=-70.0)
    neurons = LIFNeuron(n_neurons=5, config=config)

    # Corrupt state
    neurons.v.fill(0.0)
    neurons.refractory_time.fill(1.0)

    # Reset
    neurons.reset_state()

    assert np.allclose(neurons.v, -70.0)
    assert np.allclose(neurons.refractory_time, 0.0)


def test_lif_spike_generation() -> None:
    config = LIFConfig(v_rest=-70.0, v_thresh=-55.0, v_reset=-80.0, t_ref=2.0)
    neurons = LIFNeuron(n_neurons=1, config=config)

    dt = 1.0
    # Apply a huge current to guarantee a spike immediately
    current = np.array([1000.0])

    spikes = neurons.step(current, dt)

    assert spikes[0]
    # After spiking, voltage should be at v_reset
    assert np.isclose(neurons.v[0], -80.0)
    # Refractory time should be set
    assert np.isclose(neurons.refractory_time[0], 2.0)


def test_lif_refractory_period() -> None:
    config = LIFConfig(v_thresh=-55.0, v_reset=-80.0, t_ref=2.0)
    neurons = LIFNeuron(n_neurons=1, config=config)

    dt = 1.0
    # Step 1: Spike
    neurons.step(np.array([1000.0]), dt)
    assert np.isclose(neurons.refractory_time[0], 2.0)

    # Step 2: During refractory period, even huge current does not cause spike
    spikes = neurons.step(np.array([1000.0]), dt)
    assert not spikes[0]
    assert np.isclose(neurons.refractory_time[0], 1.0)
    # Voltage should remain clamped at v_reset during refraction
    assert np.isclose(neurons.v[0], -80.0)

    # Step 3: Refractory period ends
    spikes = neurons.step(np.array([1000.0]), dt)
    assert not spikes[0]
    assert np.isclose(neurons.refractory_time[0], 0.0)

    # Step 4: Ready to spike again
    spikes = neurons.step(np.array([1000.0]), dt)
    assert spikes[0]


@given(
    v_start=st.floats(min_value=-80.0, max_value=-56.0),
    current=st.floats(min_value=-10.0, max_value=10.0),
)
def test_lif_voltage_bounds(v_start: float, current: float) -> None:
    """
    Property test: voltage should never exceed threshold and remain
    stable if not spiking.
    """
    config = LIFConfig(v_rest=-70.0, v_thresh=-55.0, v_reset=-80.0, tau_m=20.0)
    neurons = LIFNeuron(n_neurons=1, config=config)

    neurons.v[0] = v_start
    dt = 1.0

    neurons.step(np.array([current]), dt)

    # Voltage must be strictly less than threshold
    assert neurons.v[0] < config.v_thresh
    # Voltage should not explode to infinity or NaN
    assert np.isfinite(neurons.v[0])


def test_lif_passive_decay() -> None:
    """Without input, voltage should decay back to v_rest."""
    config = LIFConfig(v_rest=-70.0, tau_m=20.0)
    neurons = LIFNeuron(n_neurons=1, config=config)

    # Start depolarized
    neurons.v[0] = -60.0
    dt = 1.0
    current = np.array([0.0])

    for _ in range(100):
        neurons.step(current, dt)

    # Should asymptotically reach v_rest
    assert np.isclose(neurons.v[0], -70.0, atol=0.1)
