"""Leaky Integrate-and-Fire neuron implementation."""

import numpy as np
from numba import njit  # type: ignore

from neuroflux_core.io.config import LIFConfig
from neuroflux_core.neurons.base import BaseNeuron


@njit(cache=True)  # type: ignore
def _lif_step(
    v: np.ndarray,
    refractory_time: np.ndarray,
    current: np.ndarray,
    dt: float,
    v_rest: float,
    v_thresh: float,
    v_reset: float,
    v_floor: float,
    tau_m: float,
    t_ref: float,
    g_syn: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated vectorized step function for LIF neurons.

    Membrane dynamics: tau_m * dV/dt = -(V - V_rest) + g_syn * I
    """
    n = v.shape[0]
    spikes = np.zeros(n, dtype=np.bool_)
    new_v = np.empty_like(v)
    new_refractory_time = np.empty_like(refractory_time)

    decay_factor = dt / tau_m

    for i in range(n):
        if refractory_time[i] > 0.0:
            new_refractory_time[i] = max(0.0, refractory_time[i] - dt)
            new_v[i] = v_reset
            spikes[i] = False
        else:
            v_next = v[i] + decay_factor * (-(v[i] - v_rest) + g_syn * current[i])
            if v_next < v_floor:
                v_next = v_floor
            if v_next >= v_thresh:
                spikes[i] = True
                new_v[i] = v_reset
                new_refractory_time[i] = t_ref
            else:
                spikes[i] = False
                new_v[i] = v_next
                new_refractory_time[i] = 0.0

    return new_v, new_refractory_time, spikes


class LIFNeuron(BaseNeuron):
    """
    Vectorized Leaky Integrate-and-Fire neuron group.

    Simulates N neurons simultaneously using Numba acceleration.
    """

    def __init__(self, n_neurons: int, config: LIFConfig):
        self.n_neurons = n_neurons
        self.config = config
        self.v = np.full(n_neurons, config.v_rest, dtype=np.float32)
        self.refractory_time = np.zeros(n_neurons, dtype=np.float32)

    def reset_state(self) -> None:
        self.v.fill(self.config.v_rest)
        self.refractory_time.fill(0.0)

    def step(self, current: np.ndarray, dt: float) -> np.ndarray:
        current_f32 = np.ascontiguousarray(current, dtype=np.float32)
        self.v, self.refractory_time, spikes = _lif_step(
            self.v,
            self.refractory_time,
            current_f32,
            float(dt),
            float(self.config.v_rest),
            float(self.config.v_thresh),
            float(self.config.v_reset),
            float(self.config.v_floor),
            float(self.config.tau_m),
            float(self.config.t_ref),
            float(self.config.g_syn),
        )
        return spikes  # type: ignore
