from pathlib import Path

import numpy as np
import pytest
from neuroflux_core import Network, Simulator, load_checkpoint, save_checkpoint


@pytest.fixture
def dummy_simulator() -> Simulator:
    net = Network.feedforward(layers=[5, 3], seed=0, init_scheme="uniform")
    return Simulator(net, dt=1.0, seed=0)


def test_save_and_load_checkpoint(dummy_simulator: Simulator, tmp_path: Path) -> None:
    synapse = dummy_simulator.network.synapses[0]
    synapse.weights.fill(0.77)

    filepath = tmp_path / "model.npz"
    save_checkpoint(dummy_simulator, filepath)
    assert filepath.exists()

    synapse.weights.fill(0.11)
    load_checkpoint(dummy_simulator, filepath)
    assert np.allclose(dummy_simulator.network.synapses[0].weights, 0.77)

    # reset_state must not erase loaded weights
    dummy_simulator.reset_state()
    assert np.allclose(dummy_simulator.network.synapses[0].weights, 0.77)


def test_load_checkpoint_missing_key(
    dummy_simulator: Simulator, tmp_path: Path
) -> None:
    filepath = tmp_path / "bad_model.npz"
    np.savez_compressed(filepath, some_random_key=np.array([1, 2, 3]))

    with pytest.raises(KeyError, match="missing weights for layer 0"):
        load_checkpoint(dummy_simulator, filepath)
