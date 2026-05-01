"""Population averaging and trace-based decoding."""

import numpy as np

from neuroflux_core.decoders.base import BaseDecoder


class PopulationDecoder(BaseDecoder):
    """
    Decodes signals by averaging instantaneous spike rates across a population.

    Uses exponential smoothing to trace firing rates and maps them back
    to a continuous analog scale.
    """

    def __init__(
        self,
        v_min: float,
        v_max: float,
        max_firing_rate_hz: float,
        tau_decode: float = 20.0,
    ):
        """
        Initialize the population decoder.

        Args:
            v_min: Minimum mapped analog signal value.
            v_max: Maximum mapped analog signal value.
            max_firing_rate_hz: Firing frequency mapped to v_max.
            tau_decode: Exponential smoothing time constant (ms).
        """
        self.v_min = v_min
        self.v_max = v_max
        self.max_firing_rate_hz = max_firing_rate_hz
        self.tau_decode = tau_decode
        self.filtered_rate: np.ndarray | None = None

    def reset_state(self) -> None:
        """Clear the exponential smoothing traces."""
        self.filtered_rate = None

    def decode(self, spikes: np.ndarray, dt: float) -> np.ndarray:
        """
        Decode spikes to an analog signal by population averaging.
        """
        if self.filtered_rate is None:
            self.filtered_rate = np.zeros(spikes.shape[0], dtype=np.float32)

        # Exponential decay
        decay = np.exp(-dt / self.tau_decode)

        # Add new spikes (1 spike adds 1 to the trace)
        self.filtered_rate = self.filtered_rate * decay + spikes.astype(np.float32)

        # Calculate rate in Hz based on the integration of the exponential kernel
        # Integral of exp(-t/tau) is tau, so average rate over tau is trace / tau
        rate_hz = self.filtered_rate / (self.tau_decode / 1000.0)

        # Normalize to [0, 1] relative to max_firing_rate
        normalized = rate_hz / self.max_firing_rate_hz
        normalized = np.clip(normalized, 0.0, 1.0)

        # Map to [v_min, v_max]
        decoded = self.v_min + normalized * (self.v_max - self.v_min)

        # Return the mean decoded value across the entire population as a single array
        return np.array([np.mean(decoded)], dtype=np.float32)
