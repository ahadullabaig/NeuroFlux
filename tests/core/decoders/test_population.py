import numpy as np
from neuroflux_core.decoders.population import PopulationDecoder


def test_population_decoder_zero_spikes() -> None:
    """If there are zero spikes, decoding should stick to v_min."""
    decoder = PopulationDecoder(v_min=-1.0, v_max=1.0, max_firing_rate_hz=100.0)

    dt = 1.0
    spikes = np.zeros(10, dtype=np.bool_)

    for _ in range(10):
        val = decoder.decode(spikes, dt)

    assert np.isclose(val[0], -1.0)


def test_population_decoder_max_spikes() -> None:
    """Continuous high-frequency spiking should reach and clip at v_max."""
    tau = 20.0
    max_hz = 100.0
    decoder = PopulationDecoder(
        v_min=0.0, v_max=10.0, max_firing_rate_hz=max_hz, tau_decode=tau
    )

    dt = 1.0

    # max_hz is 100Hz. If we spike 100% of the time with dt=1.0ms, that's 1000Hz!
    # Let's spike 100% of the time to ensure it clips at v_max.
    spikes = np.ones(10, dtype=np.bool_)

    # Wait for exponential to saturate (~5 tau)
    for _ in range(100):
        val = decoder.decode(spikes, dt)

    assert np.isclose(val[0], 10.0)


def test_population_decoder_convergence() -> None:
    """Test convergence to half value with 50Hz spiking when max is 100Hz."""
    tau = 50.0
    max_hz = 100.0
    decoder = PopulationDecoder(
        v_min=0.0, v_max=10.0, max_firing_rate_hz=max_hz, tau_decode=tau
    )

    dt = 1.0
    n_neurons = 10000

    # We want a 50Hz average firing rate across the population.
    # p_spike = 50 * (1 / 1000) = 0.05
    np.random.seed(42)

    # Let it run for enough time to saturate the trace
    for _ in range(200):
        spikes = np.random.rand(n_neurons) < 0.05
        val = decoder.decode(spikes, dt)

    # Expect convergence to 5.0 (half of 10.0)
    assert np.isclose(val[0], 5.0, atol=0.2)


def test_population_decoder_reset() -> None:
    decoder = PopulationDecoder(v_min=-1.0, v_max=1.0, max_firing_rate_hz=100.0)
    spikes = np.ones(10, dtype=np.bool_)
    decoder.decode(spikes, 1.0)

    assert decoder.filtered_rate is not None
    assert np.sum(decoder.filtered_rate) > 0.0

    decoder.reset_state()
    assert decoder.filtered_rate is None
