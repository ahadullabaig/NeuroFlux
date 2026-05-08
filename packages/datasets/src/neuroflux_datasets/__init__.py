"""Dataset module for NeuroFlux."""

from neuroflux_datasets.synthetic import (
    SignalGenerator,
    add_flicker,
    add_gaussian_noise,
    generate_ecg,
    generate_eeg,
    generate_sine,
)

__all__ = [
    "SignalGenerator",
    "add_flicker",
    "add_gaussian_noise",
    "generate_ecg",
    "generate_eeg",
    "generate_sine",
]
