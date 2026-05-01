"""State checkpointing utilities."""

from pathlib import Path

import numpy as np

from neuroflux_core.network.simulator import Simulator


def save_checkpoint(simulator: Simulator, filepath: str | Path) -> None:
    """
    Save the synaptic weight matrices of the simulator to a compressed .npz file.

    Args:
        simulator: The Simulator instance containing trained networks.
        filepath: The destination file path.
    """
    weights_dict = {}
    for i, synapse in enumerate(simulator.network.synapses):
        weights_dict[f"layer_{i}_weights"] = synapse.weights

    np.savez_compressed(filepath, **weights_dict)  # type: ignore


def load_checkpoint(simulator: Simulator, filepath: str | Path) -> None:
    """
    Load synaptic weight matrices from a .npz file into the simulator.

    Args:
        simulator: The Simulator instance to load weights into.
        filepath: The source .npz file path.

    Raises:
        KeyError: If the checkpoint is missing a layer's weights.
    """
    data = np.load(filepath)
    for i, synapse in enumerate(simulator.network.synapses):
        key = f"layer_{i}_weights"
        if key in data:
            synapse.weights = data[key]
            # Update the reset baseline so reset_state() doesn't erase loaded weights
            synapse.initial_weights_copy = synapse.weights.copy()
        else:
            raise KeyError(f"Checkpoint missing weights for layer {i}.")
