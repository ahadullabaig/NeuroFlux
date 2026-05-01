"""Core simulator orchestration loop."""

import numpy as np

from neuroflux_core.decoders.base import BaseDecoder
from neuroflux_core.encoders.base import BaseEncoder
from neuroflux_core.io.config import NetworkConfig
from neuroflux_core.network.feedforward import FeedForwardNetwork


class Simulator:
    """
    Orchestrates the entire execution pipeline: Encoding -> Network -> Decoding.
    """

    def __init__(
        self,
        config: NetworkConfig,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
    ):
        """
        Initialize the overarching Simulator.

        Args:
            config: Network configuration.
            encoder: Analog-to-spike translation module.
            decoder: Spike-to-analog translation module.
        """
        self.config = config
        self.network = FeedForwardNetwork(config)
        self.encoder = encoder
        self.decoder = decoder
        self.time = 0.0

    def reset_state(self) -> None:
        """Reset the entire simulator to t=0."""
        self.network.reset_state()
        self.encoder.reset_state()
        self.decoder.reset_state()
        self.time = 0.0

    def run_step(
        self, signal: np.ndarray, dt: float, training: bool = True
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Execute one complete timestep of the simulation.

        Args:
            signal: Raw analog input block for this frame.
            dt: Simulation timestep (ms).
            training: Whether to invoke STDP weight updates.

        Returns:
            Tuple of (decoded_analog_output, list_of_layer_spikes).
        """
        # 1. Encode analog signal to boolean input spikes
        input_spikes = self.encoder.encode(signal, dt)

        # 2. Propagate through feedforward network
        layer_spikes = self.network.step(input_spikes, dt, training=training)

        # 3. Decode output spikes (from the final layer) back to analog
        output_spikes = layer_spikes[-1]
        decoded_output = self.decoder.decode(output_spikes, dt)

        # Step time
        self.time += dt

        return decoded_output, layer_spikes
