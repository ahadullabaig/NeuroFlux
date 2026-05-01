import numpy as np
from neuroflux_core.io.config import STDPConfig
from neuroflux_core.synapses.stdp import STDPSynapse


def test_stdp_initialization() -> None:
    config = STDPConfig(w_min=0.1, w_max=0.9)
    # Without provided weights
    synapses = STDPSynapse(n_pre=2, n_post=3, config=config)
    assert synapses.weights.shape == (2, 3)
    assert np.all(synapses.weights >= 0.1)
    assert np.all(synapses.weights <= 0.9)
    assert np.all(synapses.trace_pre == 0.0)
    assert np.all(synapses.trace_post == 0.0)

    # With provided weights
    custom_weights = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    synapses_custom = STDPSynapse(
        n_pre=2, n_post=3, config=config, initial_weights=custom_weights
    )
    assert np.allclose(synapses_custom.weights, custom_weights)


def test_stdp_causal_spike_pair() -> None:
    """Pre-synaptic spike followed by post-synaptic spike should cause LTP."""
    config = STDPConfig(a_plus=0.1, tau_plus=10.0, w_min=0.0, w_max=1.0)
    initial_weights = np.array([[0.5]])
    synapses = STDPSynapse(
        n_pre=1, n_post=1, config=config, initial_weights=initial_weights
    )

    dt = 1.0

    # t=0: Pre spike
    synapses.step(pre_spikes=np.array([True]), post_spikes=np.array([False]), dt=dt)

    # Wait for 5 ms
    for _ in range(5):
        synapses.step(
            pre_spikes=np.array([False]), post_spikes=np.array([False]), dt=dt
        )

    # t=6: Post spike
    synapses.step(pre_spikes=np.array([False]), post_spikes=np.array([True]), dt=dt)

    # Expected weight increase:
    # trace_pre decayed for 6 steps total
    expected_trace_pre = 1.0 * np.exp(-6.0 / config.tau_plus)
    expected_weight = 0.5 + config.a_plus * expected_trace_pre

    assert np.isclose(synapses.weights[0, 0], expected_weight)
    assert synapses.weights[0, 0] > 0.5  # Weight must have increased


def test_stdp_acausal_spike_pair() -> None:
    """Post-synaptic spike followed by pre-synaptic spike should cause LTD."""
    config = STDPConfig(a_minus=0.1, tau_minus=10.0, w_min=0.0, w_max=1.0)
    initial_weights = np.array([[0.5]])
    synapses = STDPSynapse(
        n_pre=1, n_post=1, config=config, initial_weights=initial_weights
    )

    dt = 1.0

    # t=0: Post spike
    synapses.step(pre_spikes=np.array([False]), post_spikes=np.array([True]), dt=dt)

    # Wait for 5 ms
    for _ in range(5):
        synapses.step(
            pre_spikes=np.array([False]), post_spikes=np.array([False]), dt=dt
        )

    # t=6: Pre spike
    synapses.step(pre_spikes=np.array([True]), post_spikes=np.array([False]), dt=dt)

    # Expected weight decrease:
    # trace_post decayed for 6 steps total
    expected_trace_post = 1.0 * np.exp(-6.0 / config.tau_minus)
    expected_weight = 0.5 - config.a_minus * expected_trace_post

    assert np.isclose(synapses.weights[0, 0], expected_weight)
    assert synapses.weights[0, 0] < 0.5  # Weight must have decreased


def test_stdp_weight_bounds() -> None:
    """Weights should never exceed w_min or w_max."""
    config = STDPConfig(a_plus=0.5, a_minus=0.5, w_min=0.2, w_max=0.8)
    initial_weights = np.array([[0.5]])
    synapses = STDPSynapse(
        n_pre=1, n_post=1, config=config, initial_weights=initial_weights
    )

    dt = 1.0

    # Force LTP well beyond w_max
    for _ in range(10):
        # Continuous simultaneous spikes mathematically balance out in continuous time,
        # but in this trace implementation it triggers both LTP and LTD.
        # To specifically trigger LTP, we alternate Pre then Post tightly.
        synapses.step(np.array([True]), np.array([False]), dt=dt)
        synapses.step(np.array([False]), np.array([True]), dt=dt)

    assert np.isclose(synapses.weights[0, 0], 0.8)

    # Force LTD well below w_min
    for _ in range(10):
        # Alternate Post then Pre tightly to trigger LTD
        synapses.step(np.array([False]), np.array([True]), dt=dt)
        synapses.step(np.array([True]), np.array([False]), dt=dt)

    assert np.isclose(synapses.weights[0, 0], 0.2)


def test_stdp_invalid_initial_weights() -> None:
    config = STDPConfig()
    import pytest

    with pytest.raises(ValueError, match="initial_weights shape must be"):
        STDPSynapse(n_pre=2, n_post=3, config=config, initial_weights=np.array([[1.0]]))
