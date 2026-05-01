"""Step-forward (delta) encoding for continuous signals."""

import numpy as np

from neuroflux_core.encoders.base import BaseEncoder


class DeltaEncoder(BaseEncoder):
    """
    Step-forward delta encoding.

    Each per-channel baseline tracks accepted spikes. A spike fires whenever
    the input deviates from baseline by ≥ `threshold`; on a spike, that
    channel's baseline jumps to the current input value.

    Modes:
        "absolute" — fire on ``|x - baseline| ≥ threshold`` (sign-agnostic).
        "positive" — fire on ``x - baseline ≥ threshold``.
        "negative" — fire on ``x - baseline ≤ -threshold``.

    For sign-aware decoding, instantiate two encoders (one positive, one
    negative) and concatenate their outputs.

    The encoder emits no spikes on the very first call (baseline initialization).
    """

    _VALID_MODES = ("absolute", "positive", "negative")

    def __init__(self, threshold: float, mode: str = "absolute"):
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}")
        self.threshold = float(threshold)
        self.mode = mode
        self._baseline: np.ndarray | None = None

    def reset_state(self) -> None:
        self._baseline = None

    def encode(self, signal: np.ndarray, dt: float) -> np.ndarray:
        if self._baseline is None:
            self._baseline = signal.astype(np.float32).copy()
            return np.zeros(signal.shape, dtype=np.bool_)

        diff = signal - self._baseline
        if self.mode == "absolute":
            spikes = np.abs(diff) >= self.threshold
        elif self.mode == "positive":
            spikes = diff >= self.threshold
        else:  # "negative"
            spikes = diff <= -self.threshold

        self._baseline = np.where(spikes, signal.astype(np.float32), self._baseline)
        return np.asarray(spikes, dtype=np.bool_)
