"""Feedforward network architecture."""

import numpy as np

from neuroflux_core.io.config import NetworkConfig
from neuroflux_core.network.topology import BaseNetwork
from neuroflux_core.neurons.lif import LIFNeuron
from neuroflux_core.synapses.stdp import STDPSynapse


class FeedForwardNetwork(BaseNetwork):
    """
    Standard fully connected feedforward SNN architecture.
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize the feedforward network based on configuration.

        Args:
            config: NetworkConfig instance detailing layers and nested configs.
        """
        self.config = config
        self.layers = config.layers

        if len(self.layers) < 2:
            raise ValueError("Feedforward network must have at least 2 layers.")

        self.neurons = []
        self.synapses = []

        # Create hidden and output layers, and synapses between them
        for i in range(len(self.layers) - 1):
            n_pre = self.layers[i]
            n_post = self.layers[i + 1]

            # Post-synaptic neurons
            self.neurons.append(LIFNeuron(n_neurons=n_post, config=config.neuron))

            # Synapses connecting pre to post
            self.synapses.append(
                STDPSynapse(n_pre=n_pre, n_post=n_post, config=config.plasticity)
            )

    def reset_state(self) -> None:
        """Reset all neurons and synapses in the network."""
        for neuron_group in self.neurons:
            neuron_group.reset_state()

        for synapse_matrix in self.synapses:
            synapse_matrix.reset_state()

    def step(
        self, input_spikes: np.ndarray, dt: float, training: bool = True
    ) -> list[np.ndarray]:
        """
        Forward pass for one simulation timestep.

        Args:
            input_spikes: Boolean array shape (N_input,) from an encoder.
            dt: Simulation timestep in milliseconds.
            training: Whether to apply STDP learning.

        Returns:
            List of spike arrays for all layers (including input).
        """
        if input_spikes.shape[0] != self.layers[0]:
            raise ValueError("Input spikes do not match the input layer size.")

        all_spikes = [input_spikes]
        current_spikes = input_spikes

        for i in range(len(self.synapses)):
            synapse = self.synapses[i]
            neuron = self.neurons[i]

            # 1. Compute input current via dot product
            # Cast boolean spikes to float for the dot product
            # current_spikes: (N_pre,), weights: (N_pre, N_post) -> input_current:
            # (N_post,)
            input_current = np.dot(current_spikes.astype(np.float32), synapse.weights)

            # 2. Advance neurons
            post_spikes = neuron.step(input_current, dt)

            # 3. Apply plasticity
            if training:
                synapse.step(current_spikes, post_spikes, dt)

            all_spikes.append(post_spikes)
            current_spikes = post_spikes

        return all_spikes
