import numpy as np
import pytest
from neuroflux_core import (
    DeltaEncoder,
    FeedForwardNetwork,
    Network,
    NetworkConfig,
    PoissonRateEncoder,
    PopulationDecoder,
    Simulator,
)


def test_feedforward_initialization() -> None:
    config = NetworkConfig(layers=[32, 16, 8])
    net = FeedForwardNetwork(config)

    assert len(net.neurons) == 2
    assert net.neurons[0].n_neurons == 16
    assert net.neurons[1].n_neurons == 8

    assert len(net.synapses) == 2
    assert net.synapses[0].weights.shape == (32, 16)
    assert net.synapses[1].weights.shape == (16, 8)


def test_feedforward_invalid_layers() -> None:
    config = NetworkConfig(layers=[32])
    with pytest.raises(ValueError, match="at least 2 layers"):
        FeedForwardNetwork(config)


def test_feedforward_step() -> None:
    config = NetworkConfig(layers=[10, 5])
    net = FeedForwardNetwork(config)

    input_spikes = np.zeros(10, dtype=np.bool_)
    input_spikes[0] = True

    spikes = net.step(input_spikes, dt=1.0)

    assert len(spikes) == 2
    assert spikes[0].shape == (10,)
    assert spikes[1].shape == (5,)


def test_feedforward_invalid_input_size() -> None:
    config = NetworkConfig(layers=[10, 5])
    net = FeedForwardNetwork(config)

    input_spikes = np.zeros(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="Input spikes do not match"):
        net.step(input_spikes, 1.0)


def test_simulator_run_step() -> None:
    net = Network.feedforward(layers=[10, 5], seed=42)
    sim = Simulator(net, dt=1.0, seed=42)

    signal = np.full(10, 0.5)
    for _ in range(5):
        decoded, layer_spikes = sim.run_step(signal)

    assert len(layer_spikes) == 2
    assert layer_spikes[0].shape == (10,)
    assert layer_spikes[1].shape == (5,)
    assert decoded.shape == (1,)
    assert sim.time == 5.0

    sim.reset_state()
    assert sim.time == 0.0


def test_simulator_run_constant_signal() -> None:
    """run() should accept a 1D constant signal and return (T, n_out) bool."""
    net = Network.feedforward(layers=[8, 4], seed=0)
    sim = Simulator(net, dt=1.0, seed=0)

    out = sim.run(np.full(8, 0.7), duration_ms=20.0, training=False)
    assert out.shape == (20, 4)
    assert out.dtype == np.bool_


def test_simulator_run_time_varying_signal() -> None:
    net = Network.feedforward(layers=[6, 3], seed=0)
    sim = Simulator(net, dt=1.0, seed=0)
    rng = np.random.default_rng(0)
    signal = rng.uniform(0, 1, size=(50, 6))
    out = sim.run(signal, duration_ms=20.0, training=False)
    assert out.shape == (20, 3)


def test_simulator_run_signal_shape_validation() -> None:
    net = Network.feedforward(layers=[4, 2], seed=0)
    sim = Simulator(net, dt=1.0, seed=0)

    with pytest.raises(ValueError, match="1D signal length"):
        sim.run(np.zeros(7), duration_ms=10.0)
    with pytest.raises(ValueError, match="2D signal columns"):
        sim.run(np.zeros((10, 7)), duration_ms=5.0)
    with pytest.raises(ValueError, match="need"):
        sim.run(np.zeros((3, 4)), duration_ms=10.0)
    with pytest.raises(ValueError, match="signal must be 1D or 2D"):
        sim.run(np.zeros((1, 1, 1)), duration_ms=1.0)


def test_simulator_run_dt_validation() -> None:
    net = Network.feedforward(layers=[4, 2], seed=0)
    with pytest.raises(ValueError, match="dt must be positive"):
        Simulator(net, dt=0.0)
    sim = Simulator(net, dt=1.0)
    with pytest.raises(ValueError, match="duration_ms must be positive"):
        sim.run(np.zeros(4), duration_ms=0.0)


def test_simulator_decode_roundtrip() -> None:
    net = Network.feedforward(layers=[8, 4], seed=0)
    sim = Simulator(net, dt=1.0, seed=0)
    spike_train = np.zeros((30, 4), dtype=np.bool_)
    decoded = sim.decode(spike_train)
    assert decoded.shape == (30,)
    # all-zero spikes → decoder should return v_min (0.0) for default population decoder
    assert np.allclose(decoded, 0.0)


def test_simulator_decode_validates_2d() -> None:
    net = Network.feedforward(layers=[4, 2], seed=0)
    sim = Simulator(net, dt=1.0)
    with pytest.raises(ValueError, match="must be 2D"):
        sim.decode(np.zeros(10))


def test_simulator_seed_reproducibility() -> None:
    """Two Simulators with same seed must produce identical output."""

    def runit() -> tuple[np.ndarray, np.ndarray]:
        net = Network.feedforward(layers=[16, 8], seed=123, init_scheme="uniform")
        sim = Simulator(net, dt=1.0, seed=123)
        signal = np.full(16, 0.6)
        out = sim.run(signal, duration_ms=100.0, training=True)
        return out, sim.network.synapses[0].weights.copy()

    out_a, w_a = runit()
    out_b, w_b = runit()
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(w_a, w_b)


def test_simulator_different_seeds_diverge() -> None:
    def runit(seed: int) -> np.ndarray:
        net = Network.feedforward(layers=[16, 8], seed=seed, init_scheme="uniform")
        sim = Simulator(net, dt=1.0, seed=seed)
        return sim.run(np.full(16, 0.6), duration_ms=50.0, training=True)

    assert not np.array_equal(runit(1), runit(2))


def test_simulator_with_custom_encoder_decoder() -> None:
    net = Network.feedforward(layers=[6, 3], seed=0)
    rng = np.random.default_rng(0)
    enc = PoissonRateEncoder(0.0, 1.0, 100.0, rng=rng)
    dec = PopulationDecoder(0.0, 1.0, 100.0, tau_decode=10.0)
    sim = Simulator(net, dt=1.0, encoder=enc, decoder=dec)

    decoded, _ = sim.run_step(np.full(6, 0.5))
    assert decoded.shape == (1,)


def test_simulator_with_delta_encoder() -> None:
    """Delta-encoder integration: simulator should accept and propagate it."""
    net = Network.feedforward(layers=[5, 3], seed=0)
    enc = DeltaEncoder(threshold=0.1, mode="absolute")
    sim = Simulator(net, dt=1.0, encoder=enc)
    rng = np.random.default_rng(0)
    signal = rng.uniform(0, 1, size=(40, 5))
    out = sim.run(signal, duration_ms=40.0, training=False)
    assert out.shape == (40, 3)


def test_all_layers_fire_under_realistic_input() -> None:
    """End-to-end correctness: every layer must spike under nominal drive.

    This is the regression test for the silent-network bug (g_syn fix).
    """
    net = Network.feedforward(layers=[32, 16, 8], seed=42, init_scheme="uniform")
    sim = Simulator(net, dt=1.0, seed=42)
    signal = np.full(32, 0.8)

    counts = [0, 0, 0]
    for _ in range(500):
        _, layer_spikes = sim.run_step(signal, training=False)
        for i, s in enumerate(layer_spikes):
            counts[i] += int(s.sum())

    assert all(c > 0 for c in counts), (
        f"Some layers silent under realistic input: {counts}"
    )
