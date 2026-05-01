"""State checkpointing utilities."""

from pathlib import Path

import numpy as np

from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_core.network.simulator import Simulator


def _require_feedforward(simulator: Simulator) -> FeedForwardNetwork:
    if not isinstance(simulator.network, FeedForwardNetwork):
        raise TypeError(
            "Checkpointing is only supported for FeedForwardNetwork; got "
            f"{type(simulator.network).__name__}."
        )
    return simulator.network


def save_checkpoint(simulator: Simulator, filepath: str | Path) -> None:
    """
    Save the synaptic weight matrices of the simulator to a compressed .npz file.

    Args:
        simulator: The Simulator instance containing trained networks.
        filepath: The destination file path.
    """
    network = _require_feedforward(simulator)
    weights_dict: dict[str, np.ndarray] = {
        f"layer_{i}_weights": synapse.weights
        for i, synapse in enumerate(network.synapses)
    }
    np.savez_compressed(filepath, **weights_dict)  # type: ignore[arg-type]


def load_checkpoint(simulator: Simulator, filepath: str | Path) -> None:
    """
    Load synaptic weight matrices from a .npz file into the simulator.

    Args:
        simulator: The Simulator instance to load weights into.
        filepath: The source .npz file path.

    Raises:
        KeyError: If the checkpoint is missing a layer's weights.
    """
    network = _require_feedforward(simulator)
    data = np.load(filepath)
    for i, synapse in enumerate(network.synapses):
        key = f"layer_{i}_weights"
        if key in data:
            synapse.weights = data[key]
            # Update the reset baseline so reset_state() doesn't erase loaded weights
            synapse.initial_weights_copy = synapse.weights.copy()
        else:
            raise KeyError(f"Checkpoint missing weights for layer {i}.")
