import os
import time

import numpy as np
import pytest
from neuroflux_core import Network, Simulator


@pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1",
    reason="PLAN.md performance target requires JIT-compiled kernels.",
)
def test_640_synapse_performance_benchmark() -> None:
    """
    PLAN.md acceptance criterion: 640-synapse network, 10,000 timesteps,
    under 2 seconds on M-series or i7-class CPU (with Numba JIT).
    """
    net = Network.feedforward(layers=[32, 20], seed=42, init_scheme="uniform")
    sim = Simulator(net, dt=1.0, seed=42)

    rng = np.random.default_rng(42)
    signal = rng.uniform(0.0, 1.0, size=(32,))

    # Warm up Numba JIT
    sim.run_step(signal)
    sim.reset_state()

    steps = 10_000
    start = time.perf_counter()
    for _ in range(steps):
        sim.run_step(signal, training=True)
    duration = time.perf_counter() - start

    assert duration < 2.0, (
        f"Performance regression: {duration:.2f}s for 10k steps × 640 synapses "
        f"(target < 2s per PLAN.md)."
    )


def test_smoke_step_under_jit_disabled() -> None:
    """Sanity check that the step loop runs at all when JIT is off."""
    net = Network.feedforward(layers=[10, 5], seed=0)
    sim = Simulator(net, dt=1.0, seed=0)
    sim.run_step(np.full(10, 0.5))
