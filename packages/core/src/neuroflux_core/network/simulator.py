"""Core simulator orchestration loop."""

import numpy as np

from neuroflux_core.decoders.base import BaseDecoder
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.base import BaseEncoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.network.base import BaseNetwork


class Simulator:
    """
    Orchestrates the encode → network → decode pipeline.

    Reproducibility: when constructed with a `seed`, the simulator owns a
    `np.random.Generator` and wires it into the default Poisson encoder. If you
    pass a custom encoder, you are responsible for seeding its own RNG.
    """

    def __init__(
        self,
        network: BaseNetwork,
        dt: float = 1.0,
        seed: int | None = None,
        encoder: BaseEncoder | None = None,
        decoder: BaseDecoder | None = None,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.network = network
        self.dt = float(dt)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if encoder is None:
            encoder = PoissonRateEncoder(
                v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0, rng=self.rng
            )
        self.encoder = encoder

        if decoder is None:
            decoder = PopulationDecoder(
                v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0
            )
        self.decoder = decoder

        self.time = 0.0

    def reset_state(self) -> None:
        """Reset the entire simulator to t=0."""
        self.network.reset_state()
        self.encoder.reset_state()
        self.decoder.reset_state()
        self.time = 0.0

    def run_step(
        self,
        signal: np.ndarray,
        dt: float | None = None,
        training: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Execute one timestep.

        Args:
            signal: Raw analog input frame.
            dt: Override timestep (defaults to `self.dt`).
            training: Whether to apply STDP updates.

        Returns:
            (decoded_analog_output, list_of_layer_spikes_including_input).
        """
        use_dt = float(dt) if dt is not None else self.dt
        input_spikes = self.encoder.encode(signal, use_dt)
        layer_spikes = self.network.step(input_spikes, use_dt, training=training)
        decoded = self.decoder.decode(layer_spikes[-1], use_dt)
        self.time += use_dt
        return decoded, layer_spikes

    def run(
        self,
        signal: np.ndarray,
        duration_ms: float,
        training: bool = True,
    ) -> np.ndarray:
        """
        Run the simulation for `duration_ms` and return the output spike train.

        Args:
            signal: Either 1D `(n_input,)` (constant per timestep) or 2D
                `(T, n_input)` time-varying. T must be ≥ ceil(duration_ms / dt).
            duration_ms: Total simulation duration in milliseconds.
            training: Whether STDP plasticity is active.

        Returns:
            Boolean array `(n_steps, n_output)` of output-layer spikes per
            timestep.
        """
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be positive")

        n_steps = int(round(duration_ms / self.dt))
        n_input = self._first_layer_size()
        n_output = self._last_layer_size()

        sig = np.asarray(signal)
        if sig.ndim == 1:
            if sig.shape[0] != n_input:
                raise ValueError(
                    f"1D signal length {sig.shape[0]} != input layer {n_input}"
                )
            time_varying = False
        elif sig.ndim == 2:
            if sig.shape[1] != n_input:
                raise ValueError(
                    f"2D signal columns {sig.shape[1]} != input layer {n_input}"
                )
            if sig.shape[0] < n_steps:
                raise ValueError(
                    f"2D signal has {sig.shape[0]} timesteps, need {n_steps}"
                )
            time_varying = True
        else:
            raise ValueError(f"signal must be 1D or 2D, got {sig.ndim}D")

        out = np.zeros((n_steps, n_output), dtype=np.bool_)
        for t in range(n_steps):
            frame = sig[t] if time_varying else sig
            _, layer_spikes = self.run_step(frame, training=training)
            out[t] = layer_spikes[-1]
        return out

    def decode(
        self, spike_train: np.ndarray, method: str | None = None
    ) -> np.ndarray:
        """
        Apply the configured decoder to a (T, N) spike train.

        The `method` argument is accepted for forward-compatibility with the
        public API contract; the simulator's configured decoder is used. Pass a
        different decoder via `Simulator(..., decoder=...)` to change behavior.
        """
        if spike_train.ndim != 2:
            raise ValueError("spike_train must be 2D (T, N)")
        self.decoder.reset_state()
        out: list[np.ndarray] = []
        for t in range(spike_train.shape[0]):
            out.append(self.decoder.decode(spike_train[t], self.dt))
        arr = np.stack(out)
        if arr.ndim > 1 and arr.shape[-1] == 1:
            return arr.reshape(arr.shape[0])
        return arr

    def _first_layer_size(self) -> int:
        return int(self.network.layers[0])  # type: ignore[attr-defined]

    def _last_layer_size(self) -> int:
        return int(self.network.layers[-1])  # type: ignore[attr-defined]
