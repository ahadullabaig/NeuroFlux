"""Abstract base class for spike encoders."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for converting analog signals to spike trains."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal states (e.g., temporal histories)."""
        pass

    @abstractmethod
    def encode(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """
        Encode an analog signal array into a boolean spike array.

        Args:
            signal: Analog input signal array of shape (N,).
            dt: Simulation timestep in milliseconds.

        Returns:
            Boolean NumPy array of shape (N,) indicating spikes.
        """
        pass
