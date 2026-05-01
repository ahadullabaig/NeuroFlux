from pathlib import Path

import numpy as np
import pytest
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.checkpoint import load_checkpoint, save_checkpoint
from neuroflux_core.io.config import NetworkConfig
from neuroflux_core.network.simulator import Simulator


@pytest.fixture
def dummy_simulator() -> Simulator:
    config = NetworkConfig(layers=[5, 3])
    encoder = PoissonRateEncoder(0.0, 1.0, 100.0)
    decoder = PopulationDecoder(0.0, 1.0, 100.0)
    return Simulator(config, encoder, decoder)


def test_save_and_load_checkpoint(dummy_simulator: Simulator, tmp_path: Path) -> None:
    # Modify weights
    synapse = dummy_simulator.network.synapses[0]
    synapse.weights.fill(0.77)
    
    filepath = tmp_path / "model.npz"
    save_checkpoint(dummy_simulator, filepath)
    
    assert filepath.exists()
    
    # Reset to random or default
    synapse.weights.fill(0.11)
    
    # Load back
    load_checkpoint(dummy_simulator, filepath)
    
    assert np.allclose(dummy_simulator.network.synapses[0].weights, 0.77)
    # Ensure resetting doesn't break loaded weights
    dummy_simulator.reset_state()
    assert np.allclose(dummy_simulator.network.synapses[0].weights, 0.77)


def test_load_checkpoint_missing_key(
    dummy_simulator: Simulator, tmp_path: Path
) -> None:
    filepath = tmp_path / "bad_model.npz"
    np.savez_compressed(filepath, some_random_key=np.array([1, 2, 3]))
    
    with pytest.raises(KeyError, match="missing weights for layer 0"):
        load_checkpoint(dummy_simulator, filepath)
