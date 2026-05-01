"""Time-to-first-spike (latency) encoding."""

import numpy as np

from neuroflux_core.encoders.base import BaseEncoder


class TemporalEncoder(BaseEncoder):
    """
    Time-to-first-spike encoding within a fixed window.

    The encoder operates in cycles of `t_window_ms`. At the start of each
    window the instantaneous signal is sampled and per-neuron latencies are
    computed: high values fire early, low values fire late. Each neuron emits
    at most one spike per window, on the timestep its latency is reached.
    Cycles repeat for as long as the encoder is called.
    """

    def __init__(self, v_min: float, v_max: float, t_window_ms: float):
        if v_max <= v_min:
            raise ValueError("v_max must be strictly greater than v_min")
        if t_window_ms <= 0.0:
            raise ValueError("t_window_ms must be positive")
        self.v_min = v_min
        self.v_max = v_max
        self.t_window_ms = t_window_ms
        self._latency: np.ndarray | None = None
        self._fired: np.ndarray | None = None
        self._t_in_window: float = 0.0

    def reset_state(self) -> None:
        self._latency = None
        self._fired = None
        self._t_in_window = 0.0

    def encode(self, signal: np.ndarray, dt: float) -> np.ndarray:
        if self._latency is None or self._fired is None:
            normalized = (signal - self.v_min) / (self.v_max - self.v_min)
            normalized = np.clip(normalized, 0.0, 1.0)
            # Cap latency at the last timestep of the window so even the slowest
            # neuron has a chance to fire before reset.
            max_latency = max(self.t_window_ms - dt, 0.0)
            self._latency = (1.0 - normalized) * max_latency
            self._fired = np.zeros(signal.shape, dtype=np.bool_)
            self._t_in_window = 0.0

        spikes = np.asarray(
            (self._t_in_window >= self._latency) & ~self._fired, dtype=np.bool_
        )
        self._fired = self._fired | spikes
        self._t_in_window += dt

        if self._t_in_window >= self.t_window_ms:
            # Window complete; force re-sample on next call
            self._latency = None
            self._fired = None

        return spikes
