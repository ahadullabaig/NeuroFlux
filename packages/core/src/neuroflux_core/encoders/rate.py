"""Poisson rate coding implementation."""

import numpy as np

from neuroflux_core.encoders.base import BaseEncoder


class PoissonRateEncoder(BaseEncoder):
    """
    Encodes analog signals into spikes using a Poisson process.

    Higher signal values result in proportionally higher firing probabilities.
    """

    def __init__(self, v_min: float, v_max: float, max_firing_rate_hz: float):
        """
        Initialize the Poisson rate encoder.

        Args:
            v_min: Minimum expected analog signal value.
            v_max: Maximum expected analog signal value.
            max_firing_rate_hz: Firing frequency (Hz) at v_max.
        """
        self.v_min = v_min
        self.v_max = v_max
        self.max_firing_rate_hz = max_firing_rate_hz

    def reset_state(self) -> None:
        """Poisson encoders are stateless."""
        pass

    def encode(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """
        Generate spikes proportionally to the normalized signal.
        """
        # Normalize signal to [0, 1] range
        normalized = (signal - self.v_min) / (self.v_max - self.v_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Convert Hz to spikes per millisecond dt
        p_max = self.max_firing_rate_hz * (dt / 1000.0)

        # Probability of spiking per timestep
        p_spike = normalized * p_max
        p_spike = np.clip(p_spike, 0.0, 1.0)

        # Generate spikes
        return np.random.random(signal.shape) < p_spike
