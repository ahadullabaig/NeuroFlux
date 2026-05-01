"""NeuroFlux core SNN engine — public API."""

from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.decoders.spike_rate import SpikeRateDecoder
from neuroflux_core.encoders.delta import DeltaEncoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.encoders.temporal import TemporalEncoder
from neuroflux_core.io.checkpoint import load_checkpoint, save_checkpoint
from neuroflux_core.io.config import LIFConfig, NetworkConfig, STDPConfig
from neuroflux_core.network.feedforward import FeedForwardNetwork, Network
from neuroflux_core.network.simulator import Simulator
from neuroflux_core.network.topology import init_weights
from neuroflux_core.neurons.lif import LIFNeuron
from neuroflux_core.synapses.stdp import STDPSynapse

__all__ = [
    # Configs
    "LIFConfig",
    "STDPConfig",
    "NetworkConfig",
    # Network construction & execution
    "Network",
    "FeedForwardNetwork",
    "Simulator",
    "init_weights",
    # Neurons & synapses
    "LIFNeuron",
    "STDPSynapse",
    # Encoders
    "PoissonRateEncoder",
    "TemporalEncoder",
    "DeltaEncoder",
    # Decoders
    "PopulationDecoder",
    "SpikeRateDecoder",
    # I/O
    "save_checkpoint",
    "load_checkpoint",
]
