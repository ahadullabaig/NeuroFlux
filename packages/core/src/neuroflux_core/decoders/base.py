"""Abstract base class for spike decoders."""

from abc import ABC, abstractmethod

import numpy as np


class BaseDecoder(ABC):
    """Abstract base class for converting spike trains to analog signals."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal traces and histories."""
        pass

    @abstractmethod
    def decode(self, spikes: np.ndarray, dt: float) -> np.ndarray:
        """
        Decode a boolean spike array into an analog signal.

        Args:
            spikes: Boolean input spike array of shape (N,).
            dt: Simulation timestep in milliseconds.

        Returns:
            Analog NumPy array representing the decoded signal.
        """
        pass
