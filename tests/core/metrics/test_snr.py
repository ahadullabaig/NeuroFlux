import numpy as np
import pytest
from neuroflux_core.metrics.snr import calculate_snr


def test_calculate_snr_perfect_reconstruction() -> None:
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    snr = calculate_snr(signal, signal)
    assert snr == float("inf")


def test_calculate_snr_valid() -> None:
    # A simple sine wave
    t = np.linspace(0, 10, 100)
    signal_true = np.sin(t)
    
    # Add small noise
    noise = np.random.normal(0, 0.1, len(t))
    signal_pred = signal_true + noise
    
    snr = calculate_snr(signal_true, signal_pred)
    
    # Variance of sin(t) is ~0.5. Variance of noise is ~0.01.
    # SNR should be around 10 * log10(0.5 / 0.01) = 10 * log10(50) ~= 17 dB
    assert 10.0 < snr < 25.0


def test_calculate_snr_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        calculate_snr(np.array([1.0, 2.0]), np.array([1.0]))
