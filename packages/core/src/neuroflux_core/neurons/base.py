"""Abstract base class for neuron models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseNeuron(ABC):
    """Abstract base class for all vectorized neuron models."""

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the internal state variables (e.g., membrane potential)
        to default values.
        """
        pass

    @abstractmethod
    def step(self, current: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance the neuron state by one timestep `dt` given input `current`.

        Args:
            current: NumPy array of shape (N,) representing input current
                to each neuron.
            dt: The simulation timestep in milliseconds.

        Returns:
            A boolean NumPy array of shape (N,) indicating which neurons fired a spike.
        """
        pass
