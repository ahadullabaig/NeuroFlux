import numpy as np
from neuroflux_core import TemporalEncoder


def test_high_value_fires_early() -> None:
    enc = TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=10.0)
    # Two neurons: high (1.0) → latency 0; low (0.0) → latency t_window-dt = 9
    signal = np.array([1.0, 0.0])
    fire_times = [-1, -1]
    for t in range(15):
        spikes = enc.encode(signal, dt=1.0)
        for i in range(2):
            if spikes[i] and fire_times[i] == -1:
                fire_times[i] = t
    assert fire_times[0] == 0
    assert fire_times[1] == 9


def test_each_neuron_fires_once_per_window() -> None:
    enc = TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=10.0)
    signal = np.array([1.0, 0.5])
    counts = np.zeros(2, dtype=np.int64)
    for _ in range(10):  # one window
        counts += enc.encode(signal, dt=1.0).astype(np.int64)
    assert counts[0] == 1
    assert counts[1] == 1


def test_window_repeats() -> None:
    enc = TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=5.0)
    signal = np.array([1.0])
    total = 0
    for _ in range(20):  # four windows
        total += int(enc.encode(signal, dt=1.0).sum())
    # high-value neuron fires once per window → 4 spikes
    assert total == 4


def test_clipping_below_v_min() -> None:
    enc = TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=5.0)
    signal = np.array([-5.0])
    spikes_total = 0
    for _ in range(5):
        spikes_total += int(enc.encode(signal, dt=1.0).sum())
    # Below v_min clips to 0 → latency = t_window-dt = 4 → fires once at t=4
    assert spikes_total == 1


def test_reset_state_clears_window() -> None:
    enc = TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=10.0)
    enc.encode(np.array([0.5]), dt=1.0)
    enc.reset_state()
    assert enc._latency is None
    assert enc._t_in_window == 0.0


def test_invalid_bounds() -> None:
    import pytest

    with pytest.raises(ValueError):
        TemporalEncoder(v_min=1.0, v_max=1.0, t_window_ms=10.0)
    with pytest.raises(ValueError):
        TemporalEncoder(v_min=0.0, v_max=1.0, t_window_ms=0.0)
