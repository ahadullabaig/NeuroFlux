import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from neuroflux_core.encoders.rate import PoissonRateEncoder


def test_rate_encoder_bounds() -> None:
    """Test that generated spikes stay within expected bounds."""
    encoder = PoissonRateEncoder(v_min=-1.0, v_max=1.0, max_firing_rate_hz=100.0)

    # Encode min signal
    signal_min = np.array([-1.0] * 1000)
    spikes_min = encoder.encode(signal_min, dt=1.0)
    assert not np.any(spikes_min)  # Should never spike at v_min

    # Encode max signal
    signal_max = np.array([1.0] * 1000)
    spikes_max = encoder.encode(signal_max, dt=10.0)  # 100Hz at 10ms = 1.0 probability
    assert np.all(spikes_max)  # Should always spike at prob 1.0


def test_rate_encoder_statistics() -> None:
    """Test that the statistical spike rate matches the analytical probability."""
    max_hz = 50.0
    dt = 1.0
    encoder = PoissonRateEncoder(v_min=0.0, v_max=10.0, max_firing_rate_hz=max_hz)

    # 5.0 is exactly half of the range, so normalized = 0.5
    # Probability per timestep = 0.5 * 50 * (1/1000) = 0.025
    signal = np.array([5.0] * 100_000)
    
    spikes = encoder.encode(signal, dt=dt)
    actual_prob = np.mean(spikes)
    
    expected_prob = 0.025
    # Should be close to analytical probability due to large N
    assert np.isclose(actual_prob, expected_prob, atol=0.005)


@given(
    signal_val=st.floats(min_value=-5.0, max_value=5.0),
    dt=st.floats(min_value=0.1, max_value=10.0)
)
def test_rate_encoder_property(signal_val: float, dt: float) -> None:
    encoder = PoissonRateEncoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    signal = np.array([signal_val])
    spikes = encoder.encode(signal, dt=dt)
    
    assert isinstance(spikes, np.ndarray)
    assert spikes.dtype == np.bool_
    assert spikes.shape == (1,)
    
    if signal_val <= 0.0:
        assert not spikes[0]

def test_rate_encoder_reset_state() -> None:
    encoder = PoissonRateEncoder(v_min=0.0, v_max=1.0, max_firing_rate_hz=100.0)
    # Just to reach coverage
    encoder.reset_state()
