"""Synthetic signal and noise generators for Phase 2 validation."""

from typing import Any, Literal

import numpy as np


def generate_sine(
    duration_s: float,
    fs: float,
    freq: float = 2.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a pure sinusoidal reference signal."""
    t = np.arange(int(duration_s * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_eeg(
    duration_s: float,
    fs: float,
    bands: list[tuple[float, float]] | None = None,
    spike_rate_hz: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate an EEG-like signal via superposition of sinusoidal bands.

    Args:
        duration_s: Duration in seconds.
        fs: Sampling frequency in Hz.
        bands: List of (frequency_hz, amplitude) tuples.
            Defaults to alpha, beta, gamma.
        spike_rate_hz: Rate of epileptiform spike injection per second.
        rng: Random number generator.
    """
    if bands is None:
        bands = [(10.0, 1.0), (20.0, 0.5), (40.0, 0.25)]

    gen = rng if rng is not None else np.random.default_rng()
    t = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(t)

    for freq_hz, amp in bands:
        phase_shift = gen.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq_hz * t + phase_shift)

    if spike_rate_hz > 0:
        n_spikes = gen.poisson(spike_rate_hz * duration_s)
        spike_times = gen.uniform(0, duration_s, size=n_spikes)
        for st in spike_times:
            # Typical epileptiform spike: high amplitude, fast transient
            signal += 5.0 * np.exp(-((t - st) ** 2) / (2 * 0.005**2))

    return signal


def generate_ecg(
    duration_s: float,
    fs: float,
    heart_rate: float = 60.0,
    p_multiplier: float = 1.0,
    baseline_wander_amplitude: float = 0.0,
) -> np.ndarray:
    """
    Generate an ECG-like signal with QRS complex morphology.

    Args:
        duration_s: Duration in seconds.
        fs: Sampling frequency in Hz.
        heart_rate: Simulated heart rate in BPM.
        p_multiplier: Multiplier for P-wave amplitude.
        baseline_wander_amplitude: Amplitude of 0.5Hz baseline wander.
    """
    t = np.arange(int(duration_s * fs)) / fs
    beat_interval = 60.0 / heart_rate

    # Center each beat window around 0
    beat_t = (t % beat_interval) - (0.5 * beat_interval)

    # Parameters for Gaussian sum: (amplitude, center_offset, width)
    waves = [
        (0.15 * p_multiplier, -0.2, 0.02),  # P wave
        (-0.15, -0.05, 0.01),  # Q wave
        (1.0, 0.0, 0.015),  # R wave
        (-0.25, 0.05, 0.01),  # S wave
        (0.3, 0.3, 0.04),  # T wave
    ]

    signal = np.zeros_like(t)
    for a, b, c in waves:
        signal += a * np.exp(-((beat_t - b) ** 2) / (2 * c**2))

    if baseline_wander_amplitude > 0:
        signal += baseline_wander_amplitude * np.sin(2 * np.pi * 0.5 * t)

    return signal


def add_flicker(
    signal: np.ndarray, fs: float, freq: float, amplitude: float
) -> np.ndarray:
    """Add periodic flicker interference."""
    t = np.arange(len(signal)) / fs
    return signal + amplitude * np.sin(2 * np.pi * freq * t)


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian white noise to achieve a target SNR."""
    gen = rng if rng is not None else np.random.default_rng()

    sig_power = np.mean(signal**2)
    if sig_power == 0:
        return signal.copy()

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = gen.normal(0, np.sqrt(noise_power), size=len(signal))
    return signal + noise


class SignalGenerator:
    """Unified pipeline API for composing clean and noisy signals."""

    def __init__(self, fs: float = 250.0, seed: int | None = None):
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        signal_type: Literal["ecg", "eeg", "sine"],
        duration_s: float,
        flicker_freq: float | None = None,
        flicker_amplitude: float = 0.0,
        snr_db: float | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic signal and its noisy counterpart.

        Args:
            signal_type: "ecg", "eeg", or "sine".
            duration_s: Signal duration in seconds.
            flicker_freq: Hz of the periodic interference.
            flicker_amplitude: Amplitude of the flicker.
            snr_db: Target SNR for Gaussian noise (None to disable).
            **kwargs: Extra params for the specific generator.

        Returns:
            Tuple of (clean_signal, noisy_signal)
        """
        if signal_type == "ecg":
            clean = generate_ecg(duration_s, self.fs, **kwargs)
        elif signal_type == "eeg":
            clean = generate_eeg(duration_s, self.fs, rng=self.rng, **kwargs)
        elif signal_type == "sine":
            clean = generate_sine(duration_s, self.fs, **kwargs)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        noisy = clean.copy()

        if flicker_freq is not None and flicker_amplitude > 0:
            noisy = add_flicker(
                noisy, self.fs, float(flicker_freq), float(flicker_amplitude)
            )

        if snr_db is not None:
            noisy = add_gaussian_noise(noisy, float(snr_db), self.rng)

        return clean, noisy
