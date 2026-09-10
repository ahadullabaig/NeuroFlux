"""Feedforward network architecture."""

import numpy as np

from neuroflux_core.io.config import LIFConfig, NetworkConfig, STDPConfig
from neuroflux_core.network.base import BaseNetwork
from neuroflux_core.network.topology import InitScheme, init_weights
from neuroflux_core.neurons.lif import LIFNeuron
from neuroflux_core.synapses.stdp import STDPSynapse


class FeedForwardNetwork(BaseNetwork):
    """Standard fully connected feedforward SNN architecture."""

    @property
    def synapses(self) -> list[STDPSynapse]:
        return self._synapses

    def __init__(
        self,
        config: NetworkConfig,
        rng: np.random.Generator | None = None,
        init_scheme: InitScheme = "uniform",
    ):
        """
        Args:
            config: Network configuration (layers, neuron, plasticity).
            rng: Optional Generator for reproducible weight initialization.
            init_scheme: Weight init scheme — see `network.topology.init_weights`.
        """
        self.config = config
        self.layers = config.layers

        if len(self.layers) < 2:
            raise ValueError("Feedforward network must have at least 2 layers.")

        self._rng = rng if rng is not None else np.random.default_rng()
        self.init_scheme = init_scheme

        self.neurons: list[LIFNeuron] = []
        self._synapses: list[STDPSynapse] = []

        for i in range(len(self.layers) - 1):
            n_pre = self.layers[i]
            n_post = self.layers[i + 1]

            self.neurons.append(
                LIFNeuron(
                    n_neurons=n_post,
                    config=config.neuron.model_copy(
                        update={"g_syn": config.gain_for_layer(i)}
                    ),
                )
            )

            initial = init_weights(
                shape=(n_pre, n_post),
                scheme=init_scheme,
                rng=self._rng,
                w_min=config.plasticity.w_min,
                w_max=config.plasticity.w_max,
            )
            self._synapses.append(
                STDPSynapse(
                    n_pre=n_pre,
                    n_post=n_post,
                    config=config.plasticity,
                    initial_weights=initial,
                    rng=self._rng,
                )
            )

    @classmethod
    def build(
        cls,
        layers: list[int],
        neuron: LIFConfig | None = None,
        plasticity: STDPConfig | None = None,
        seed: int | None = None,
        init_scheme: InitScheme = "uniform",
        layer_gains: list[float] | None = None,
    ) -> "FeedForwardNetwork":
        """Convenience constructor matching the public API contract."""
        config = NetworkConfig(
            layers=layers,
            neuron=neuron if neuron is not None else LIFConfig(),
            plasticity=plasticity if plasticity is not None else STDPConfig(),
            layer_gains=layer_gains,
        )
        return cls(
            config,
            rng=np.random.default_rng(seed),
            init_scheme=init_scheme,
        )

    def set_layer_gains(self, gains: list[float]) -> None:
        """
        Override the synaptic gain of each neuron group in place.

        Rate calibration needs to vary gain per layer without rebuilding the
        network and losing its weights. Updates `self.config.layer_gains` so a
        later `save_checkpoint`/`NetworkConfig` round-trip stays faithful.

        Args:
            gains: One strictly positive gain per neuron group.

        Raises:
            ValueError: If the count is wrong or any gain is non-positive.
        """
        if len(gains) != len(self.neurons):
            raise ValueError(f"Expected {len(self.neurons)} gains, got {len(gains)}.")
        if any(g <= 0.0 for g in gains):
            raise ValueError("Gains must be strictly positive.")

        for neuron_group, gain in zip(self.neurons, gains, strict=True):
            neuron_group.config = neuron_group.config.model_copy(
                update={"g_syn": float(gain)}
            )
        self.config = self.config.model_copy(
            update={"layer_gains": [float(g) for g in gains]}
        )

    @property
    def layer_gains(self) -> list[float]:
        """The synaptic gain currently in force for each neuron group."""
        return [n.config.g_syn for n in self.neurons]

    def reset_state(self) -> None:
        for neuron_group in self.neurons:
            neuron_group.reset_state()
        for synapse_matrix in self._synapses:
            synapse_matrix.reset_state()

    def step(
        self, input_spikes: np.ndarray, dt: float, training: bool = True
    ) -> list[np.ndarray]:
        if input_spikes.shape[0] != self.layers[0]:
            raise ValueError("Input spikes do not match the input layer size.")

        all_spikes: list[np.ndarray] = [input_spikes]
        current_spikes = input_spikes

        for i in range(len(self._synapses)):
            synapse = self._synapses[i]
            neuron = self.neurons[i]

            input_current = np.dot(current_spikes.astype(np.float32), synapse.weights)
            post_spikes = neuron.step(input_current, dt)

            if training:
                synapse.step(current_spikes, post_spikes, dt)

            all_spikes.append(post_spikes)
            current_spikes = post_spikes

        return all_spikes


class Network:
    """Public namespace for network construction factories."""

    @staticmethod
    def feedforward(
        layers: list[int],
        neuron: LIFConfig | None = None,
        plasticity: STDPConfig | None = None,
        seed: int | None = None,
        init_scheme: InitScheme = "uniform",
        layer_gains: list[float] | None = None,
    ) -> FeedForwardNetwork:
        """Construct a feedforward SNN matching the `Network.feedforward(...)` API."""
        return FeedForwardNetwork.build(
            layers=layers,
            neuron=neuron,
            plasticity=plasticity,
            seed=seed,
            init_scheme=init_scheme,
            layer_gains=layer_gains,
        )
