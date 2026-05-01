import time

import numpy as np
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.config import NetworkConfig
from neuroflux_core.network.simulator import Simulator


def test_640_synapse_performance_benchmark() -> None:
    """
    Performance benchmark running a dense topology of 640 synapses.
    The benchmark verifies that 100 timesteps evaluate extremely fast.
    """
    # 32 input * 20 hidden = 640 synapses. (Plus some output layer if needed, 
    # but the instructions specifically mention 640 synapses, so we use [32, 20])
    config = NetworkConfig(layers=[32, 20])
    encoder = PoissonRateEncoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    decoder = PopulationDecoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    
    sim = Simulator(config, encoder, decoder)
    
    dt = 1.0
    signal = np.random.uniform(0.0, 1.0, size=(32,))
    
    # Warmup for Numba JIT compilation
    sim.run_step(signal, dt)
    sim.reset_state()
    
    # Run Benchmark
    steps = 1000
    start_time = time.perf_counter()
    
    for _ in range(steps):
        # We can dynamically vary the signal or keep it static
        sim.run_step(signal, dt, training=True)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # For a small network compiled with Numba, 1000 steps should take < 0.1 
    # seconds usually.
    # We assert a highly generous upper bound to prevent CI flakes while 
    # ensuring no massive regressions.
    assert duration < 5.0, f"Performance heavily degraded: {duration:.2f} seconds."
