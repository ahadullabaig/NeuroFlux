"""Abstract base class for synapse models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSynapse(ABC):
    """Abstract base class for all vectorized synapse models."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset weights to their initial values and clear traces."""
        pass

    @abstractmethod
    def step(
        self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float
    ) -> None:
        """
        Update synaptic state given pre- and post-synaptic spikes.

        Args:
            pre_spikes: Boolean array of shape (N_pre,) indicating pre-synaptic spikes.
            post_spikes: Boolean array of shape (N_post,) indicating post-synaptic
                spikes.
            dt: The simulation timestep in milliseconds.
        """
        pass
