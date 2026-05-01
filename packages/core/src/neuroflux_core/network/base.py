"""Abstract base class for network topologies."""

from abc import ABC, abstractmethod
from typing import Any

from typing import Any

import numpy as np


class BaseNetwork(ABC):
    """Abstract base class for spiking neural networks."""

    @property
    @abstractmethod
    def synapses(self) -> list[Any]:
        """Return the list of synapse instances in the network."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the internal states of all neurons and synapses."""
        pass

    @abstractmethod
    def step(
        self, input_spikes: np.ndarray, dt: float, training: bool = True
    ) -> list[np.ndarray]:
        """
        Advance the network state by one timestep.

        Args:
            input_spikes: Boolean array of input spikes.
            dt: Timestep in milliseconds.
            training: If True, synaptic plasticity is enabled.

        Returns:
            A list of boolean arrays representing spikes at each layer, starting
            with the input layer.
        """
        pass
