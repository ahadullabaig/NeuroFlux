import numpy as np
import pytest
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.config import NetworkConfig
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_core.network.simulator import Simulator


def test_feedforward_initialization() -> None:
    config = NetworkConfig(layers=[32, 16, 8])
    net = FeedForwardNetwork(config)
    
    assert len(net.neurons) == 2
    assert net.neurons[0].n_neurons == 16
    assert net.neurons[1].n_neurons == 8
    
    assert len(net.synapses) == 2
    assert net.synapses[0].weights.shape == (32, 16)
    assert net.synapses[1].weights.shape == (16, 8)


def test_feedforward_invalid_layers() -> None:
    config = NetworkConfig(layers=[32])
    with pytest.raises(ValueError, match="at least 2 layers"):
        FeedForwardNetwork(config)


def test_feedforward_step() -> None:
    config = NetworkConfig(layers=[10, 5])
    net = FeedForwardNetwork(config)
    
    input_spikes = np.zeros(10, dtype=np.bool_)
    input_spikes[0] = True
    
    dt = 1.0
    spikes = net.step(input_spikes, dt)
    
    assert len(spikes) == 2
    assert spikes[0].shape == (10,)
    assert spikes[1].shape == (5,)


def test_feedforward_invalid_input_size() -> None:
    config = NetworkConfig(layers=[10, 5])
    net = FeedForwardNetwork(config)
    
    input_spikes = np.zeros(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="Input spikes do not match"):
        net.step(input_spikes, 1.0)


def test_simulator_integration() -> None:
    config = NetworkConfig(layers=[10, 5])
    encoder = PoissonRateEncoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    decoder = PopulationDecoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    
    sim = Simulator(config, encoder, decoder)
    
    # Run a few steps
    signal = np.array([0.5] * 10)
    dt = 1.0
    
    for _ in range(5):
        decoded, layer_spikes = sim.run_step(signal, dt)
        
    assert len(layer_spikes) == 2
    assert layer_spikes[0].shape == (10,)
    assert layer_spikes[1].shape == (5,)
    assert decoded.shape == (1,)
    assert sim.time == 5.0
    
    # Test reset
    sim.reset_state()
    assert sim.time == 0.0
