"""Spike-Timing-Dependent Plasticity (STDP) implementation."""

import numpy as np
from numba import njit  # type: ignore

from neuroflux_core.io.config import STDPConfig
from neuroflux_core.synapses.base import BaseSynapse


@njit(cache=True)  # type: ignore
def _stdp_step(
    weights: np.ndarray,
    trace_pre: np.ndarray,
    trace_post: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    dt: float,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
    w_min: float,
    w_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated STDP step using O(1) eligibility traces.
    """
    decay_pre = np.exp(-dt / tau_plus)
    decay_post = np.exp(-dt / tau_minus)

    # 1. Decay traces from previous timesteps
    new_trace_pre = trace_pre * decay_pre
    new_trace_post = trace_post * decay_post

    n_pre = weights.shape[0]
    n_post = weights.shape[1]

    new_weights = np.empty_like(weights)

    # 2. Weight updates
    for i in range(n_pre):
        for j in range(n_post):
            w = weights[i, j]

            # LTD: Pre spike interacts with the existing post trace
            if pre_spikes[i]:
                w -= a_minus * new_trace_post[j]

            # LTP: Post spike interacts with the existing pre trace
            if post_spikes[j]:
                w += a_plus * new_trace_pre[i]

            # Hard clamp weights
            if w < w_min:
                w = w_min
            elif w > w_max:
                w = w_max

            new_weights[i, j] = w

    # 3. Add current spikes to the traces for future interactions
    for i in range(n_pre):
        if pre_spikes[i]:
            new_trace_pre[i] += 1.0

    for j in range(n_post):
        if post_spikes[j]:
            new_trace_post[j] += 1.0

    return new_weights, new_trace_pre, new_trace_post


class STDPSynapse(BaseSynapse):
    """
    Spike-Timing-Dependent Plasticity learning rule.

    Modulates a weight matrix connecting N_pre to N_post neurons.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: STDPConfig,
        initial_weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the STDP synapses.

        Args:
            n_pre: Number of pre-synaptic neurons.
            n_post: Number of post-synaptic neurons.
            config: Configuration parameters for STDP.
            initial_weights: Optional starting weights. If None, randomly initialized.
            rng: Optional Generator for reproducible weight initialization.
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config

        if initial_weights is not None:
            if initial_weights.shape != (n_pre, n_post):
                raise ValueError("initial_weights shape must be (n_pre, n_post)")
            self.weights = initial_weights.astype(np.float32).copy()
        else:
            gen = rng if rng is not None else np.random.default_rng()
            self.weights = gen.uniform(
                config.w_min, config.w_max, size=(n_pre, n_post)
            ).astype(np.float32)

        self.initial_weights_copy = self.weights.copy()
        self.trace_pre = np.zeros(n_pre, dtype=np.float32)
        self.trace_post = np.zeros(n_post, dtype=np.float32)

    def reset_state(self) -> None:
        """Reset weights to their initialized values and clear traces."""
        self.weights = self.initial_weights_copy.copy()
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)

    def step(
        self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float
    ) -> None:
        """Advance the synaptic states by one timestep."""
        self.weights, self.trace_pre, self.trace_post = _stdp_step(
            self.weights,
            self.trace_pre,
            self.trace_post,
            pre_spikes,
            post_spikes,
            float(dt),
            float(self.config.a_plus),
            float(self.config.a_minus),
            float(self.config.tau_plus),
            float(self.config.tau_minus),
            float(self.config.w_min),
            float(self.config.w_max),
        )
