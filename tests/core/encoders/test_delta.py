import numpy as np
import pytest
from neuroflux_core import DeltaEncoder


def test_first_call_no_spikes() -> None:
    enc = DeltaEncoder(threshold=0.1)
    spikes = enc.encode(np.array([0.5, 1.0, 0.0]), dt=1.0)
    assert not np.any(spikes)


def test_absolute_mode_fires_on_change() -> None:
    enc = DeltaEncoder(threshold=0.1, mode="absolute")
    enc.encode(np.array([0.0, 0.0, 0.0]), dt=1.0)  # baseline = [0,0,0]
    spikes = enc.encode(np.array([0.2, -0.2, 0.05]), dt=1.0)
    assert spikes.tolist() == [True, True, False]


def test_positive_mode_only_positive_deltas() -> None:
    enc = DeltaEncoder(threshold=0.1, mode="positive")
    enc.encode(np.array([0.0, 0.0]), dt=1.0)
    spikes = enc.encode(np.array([0.2, -0.2]), dt=1.0)
    assert spikes.tolist() == [True, False]


def test_negative_mode_only_negative_deltas() -> None:
    enc = DeltaEncoder(threshold=0.1, mode="negative")
    enc.encode(np.array([0.0, 0.0]), dt=1.0)
    spikes = enc.encode(np.array([0.2, -0.2]), dt=1.0)
    assert spikes.tolist() == [False, True]


def test_baseline_jumps_on_spike() -> None:
    enc = DeltaEncoder(threshold=0.1, mode="absolute")
    enc.encode(np.array([0.0]), dt=1.0)  # baseline = 0
    enc.encode(np.array([0.2]), dt=1.0)  # spike → baseline = 0.2
    spikes = enc.encode(np.array([0.25]), dt=1.0)  # |0.25 - 0.2| = 0.05 < 0.1
    assert not spikes[0]
    spikes = enc.encode(np.array([0.35]), dt=1.0)  # |0.35 - 0.2| = 0.15 ≥ 0.1
    assert spikes[0]


def test_reset_state_clears_baseline() -> None:
    enc = DeltaEncoder(threshold=0.1)
    enc.encode(np.array([0.5]), dt=1.0)
    enc.reset_state()
    spikes = enc.encode(np.array([10.0]), dt=1.0)
    # First-call after reset → no spikes
    assert not np.any(spikes)


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        DeltaEncoder(threshold=0.0)


def test_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        DeltaEncoder(threshold=0.1, mode="bogus")
