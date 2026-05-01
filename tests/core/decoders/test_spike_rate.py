import numpy as np
import pytest
from neuroflux_core import SpikeRateDecoder


def test_zero_spikes_decode_to_v_min() -> None:
    dec = SpikeRateDecoder(v_min=-1.0, v_max=1.0, max_firing_rate_hz=100.0)
    spikes = np.zeros(10, dtype=np.bool_)
    val = -2.0
    for _ in range(20):
        val = dec.decode(spikes, dt=1.0)[0]
    assert val == -1.0


def test_full_spikes_clip_to_v_max() -> None:
    dec = SpikeRateDecoder(
        v_min=0.0, v_max=10.0, max_firing_rate_hz=100.0, window_ms=20.0
    )
    spikes = np.ones(10, dtype=np.bool_)
    val = 0.0
    for _ in range(30):
        val = dec.decode(spikes, dt=1.0)[0]
    # Constant firing 100% per ms = 1000 Hz, far above max → clip to v_max
    assert val == 10.0


def test_window_slides() -> None:
    dec = SpikeRateDecoder(
        v_min=0.0, v_max=10.0, max_firing_rate_hz=100.0, window_ms=10.0
    )
    spikes_on = np.ones(5, dtype=np.bool_)
    spikes_off = np.zeros(5, dtype=np.bool_)
    # Saturate window with spikes
    for _ in range(20):
        dec.decode(spikes_on, dt=1.0)
    val_on = dec.decode(spikes_on, dt=1.0)[0]
    # Drain with zeros — value should drop
    for _ in range(20):
        last_off = dec.decode(spikes_off, dt=1.0)[0]
    assert last_off < val_on
    assert last_off == 0.0


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="v_max must be"):
        SpikeRateDecoder(v_min=1.0, v_max=1.0, max_firing_rate_hz=100.0)
    with pytest.raises(ValueError, match="max_firing_rate"):
        SpikeRateDecoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=0.0)
    with pytest.raises(ValueError, match="window_ms"):
        SpikeRateDecoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0, window_ms=0.0)


def test_reset_state() -> None:
    dec = SpikeRateDecoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    dec.decode(np.ones(5, dtype=np.bool_), dt=1.0)
    assert len(dec._spike_buf) == 1
    dec.reset_state()
    assert len(dec._spike_buf) == 0
    assert dec._total == 0.0
