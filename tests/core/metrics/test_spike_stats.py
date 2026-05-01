import numpy as np
import pytest
from neuroflux_core.metrics.spike_stats import (
    fano_factor,
    mean_firing_rate,
    silent_neuron_fraction,
)


def test_mean_firing_rate_basic() -> None:
    # 1000 timesteps, dt=1ms → 1 second total
    # neuron 0: spikes every other step → 500 Hz
    # neuron 1: never spikes → 0 Hz
    spikes = np.zeros((1000, 2), dtype=np.bool_)
    spikes[::2, 0] = True
    rates = mean_firing_rate(spikes, dt=1.0)
    assert np.isclose(rates[0], 500.0)
    assert rates[1] == 0.0


def test_mean_firing_rate_validates() -> None:
    with pytest.raises(ValueError, match="2D"):
        mean_firing_rate(np.zeros(10, dtype=np.bool_), dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        mean_firing_rate(np.zeros((10, 2), dtype=np.bool_), dt=0.0)


def test_silent_neuron_fraction() -> None:
    # 4 neurons, 2 silent
    spikes = np.zeros((100, 4), dtype=np.bool_)
    spikes[5, 0] = True
    spikes[10, 2] = True
    assert silent_neuron_fraction(spikes) == 0.5


def test_silent_neuron_fraction_empty_neurons() -> None:
    assert silent_neuron_fraction(np.zeros((10, 0), dtype=np.bool_)) == 0.0


def test_fano_factor_poisson_is_near_one() -> None:
    """For Poisson spiking, Fano ≈ 1."""
    rng = np.random.default_rng(0)
    rate = 0.05  # spikes/ms
    spikes = (rng.random((10000, 50)) < rate).astype(np.bool_)
    ff = fano_factor(spikes, dt=1.0, window_ms=100.0)
    # Should hover near 1 for true Poisson; loose bounds for sample variance
    assert 0.7 < ff.mean() < 1.3


def test_fano_factor_silent_neurons_zero() -> None:
    spikes = np.zeros((100, 5), dtype=np.bool_)
    ff = fano_factor(spikes, dt=1.0, window_ms=10.0)
    assert np.all(ff == 0.0)


def test_fano_factor_short_train_returns_zeros() -> None:
    spikes = np.ones((5, 3), dtype=np.bool_)
    ff = fano_factor(spikes, dt=1.0, window_ms=100.0)
    assert ff.shape == (3,)
    assert np.all(ff == 0.0)


def test_fano_factor_validates() -> None:
    with pytest.raises(ValueError, match="2D"):
        fano_factor(np.zeros(10, dtype=np.bool_), dt=1.0, window_ms=10.0)
    with pytest.raises(ValueError, match="positive"):
        fano_factor(
            np.zeros((10, 2), dtype=np.bool_), dt=1.0, window_ms=0.0
        )
