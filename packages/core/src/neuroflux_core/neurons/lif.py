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
    tau_m: float,
    t_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated vectorized step function for LIF neurons.

    Args:
        v: Current membrane potentials.
        refractory_time: Remaining refractory time for each neuron.
        current: Input current for this timestep.
        dt: Simulation timestep (ms).
        v_rest: Resting potential (mV).
        v_thresh: Spiking threshold (mV).
        v_reset: Reset potential (mV).
        tau_m: Membrane time constant (ms).
        t_ref: Absolute refractory period (ms).

    Returns:
        Tuple of (new_v, new_refractory_time, spikes).
    """
    n = v.shape[0]
    spikes = np.zeros(n, dtype=np.bool_)
    new_v = np.empty_like(v)
    new_refractory_time = np.empty_like(refractory_time)

    decay_factor = dt / tau_m

    for i in range(n):
        if refractory_time[i] > 0.0:
            # Neuron is in refractory period
            new_refractory_time[i] = max(0.0, refractory_time[i] - dt)
            new_v[i] = v_reset
            spikes[i] = False
        else:
            # Update voltage using Euler integration
            # Membrane dynamics: tau_m * dV/dt = -(V - V_rest) + I
            v_next = v[i] + decay_factor * (-(v[i] - v_rest) + current[i])

            if v_next >= v_thresh:
                # Spike generated
                spikes[i] = True
                new_v[i] = v_reset
                new_refractory_time[i] = t_ref
            else:
                # No spike, just update voltage
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
        """
        Initialize the LIF neuron group.

        Args:
            n_neurons: Number of neurons in this group.
            config: Configuration parameters for the LIF model.
        """
        self.n_neurons = n_neurons
        self.config = config

        self.v = np.full(n_neurons, config.v_rest, dtype=np.float32)
        self.refractory_time = np.zeros(n_neurons, dtype=np.float32)

    def reset_state(self) -> None:
        """Reset membrane potential to rest and clear refractory periods."""
        self.v.fill(self.config.v_rest)
        self.refractory_time.fill(0.0)

    def step(self, current: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance the neuron states by one timestep.

        Args:
            current: NumPy array of shape (N,) representing input current.
            dt: Simulation timestep in milliseconds.

        Returns:
            Boolean array of shape (N,) indicating spikes.
        """
        self.v, self.refractory_time, spikes = _lif_step(
            self.v,
            self.refractory_time,
            current,
            float(dt),
            float(self.config.v_rest),
            float(self.config.v_thresh),
            float(self.config.v_reset),
            float(self.config.tau_m),
            float(self.config.t_ref),
        )
        return spikes  # type: ignore
