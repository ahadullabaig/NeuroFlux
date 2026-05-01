"""Sliding-window spike-rate decoding."""

from collections import deque

import numpy as np

from neuroflux_core.decoders.base import BaseDecoder


class SpikeRateDecoder(BaseDecoder):
    """
    Sliding-window mean firing rate decoder.

    Counts spikes within the trailing `window_ms`, converts to mean per-neuron
    firing rate (Hz), normalizes against `max_firing_rate_hz`, then maps to
    `[v_min, v_max]`.

    Returns shape (1,) — the population-mean decoded value at the current time.
    """

    def __init__(
        self,
        v_min: float,
        v_max: float,
        max_firing_rate_hz: float,
        window_ms: float = 50.0,
    ):
        if v_max <= v_min:
            raise ValueError("v_max must be strictly greater than v_min")
        if max_firing_rate_hz <= 0.0:
            raise ValueError("max_firing_rate_hz must be positive")
        if window_ms <= 0.0:
            raise ValueError("window_ms must be positive")
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.max_firing_rate_hz = float(max_firing_rate_hz)
        self.window_ms = float(window_ms)
        self._spike_buf: deque[np.ndarray] = deque()
        self._dt_buf: deque[float] = deque()
        self._total: float = 0.0

    def reset_state(self) -> None:
        self._spike_buf.clear()
        self._dt_buf.clear()
        self._total = 0.0

    def decode(self, spikes: np.ndarray, dt: float) -> np.ndarray:
        self._spike_buf.append(spikes.astype(np.float32))
        self._dt_buf.append(float(dt))
        self._total += float(dt)

        while self._total > self.window_ms and len(self._spike_buf) > 1:
            self._spike_buf.popleft()
            self._total -= self._dt_buf.popleft()

        n_spikes = float(sum(s.sum() for s in self._spike_buf))
        n_neurons = float(spikes.shape[0]) if spikes.shape[0] > 0 else 1.0
        elapsed = max(self._total, dt)
        rate_hz = (n_spikes / elapsed) * 1000.0 / n_neurons

        normalized = min(rate_hz / self.max_firing_rate_hz, 1.0)
        decoded = self.v_min + normalized * (self.v_max - self.v_min)
        return np.array([decoded], dtype=np.float32)
