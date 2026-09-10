"""Configuration loading and validation."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LIFConfig(BaseModel):
    """Configuration for Leaky Integrate-and-Fire neurons."""

    model_config = ConfigDict(extra="forbid")

    v_rest: float = Field(default=-70.0, description="Resting membrane potential (mV)")
    v_thresh: float = Field(default=-55.0, description="Spiking threshold (mV)")
    v_reset: float = Field(
        default=-80.0, description="Reset potential after spike (mV)"
    )
    v_floor: float = Field(
        default=-90.0, description="Hyperpolarization floor (mV); voltage clamped here"
    )
    tau_m: float = Field(default=20.0, description="Membrane time constant (ms)")
    t_ref: float = Field(default=2.0, description="Absolute refractory period (ms)")
    g_syn: float = Field(
        default=30.0,
        description="Synaptic gain — scales summed input current to mV-equivalent",
    )


class STDPConfig(BaseModel):
    """Configuration for Spike-Timing-Dependent Plasticity."""

    model_config = ConfigDict(extra="forbid")

    a_plus: float = Field(default=0.01, description="LTP learning rate (A+)")
    a_minus: float = Field(default=0.012, description="LTD learning rate (A-)")
    tau_plus: float = Field(default=20.0, description="LTP time constant (ms)")
    tau_minus: float = Field(default=20.0, description="LTD time constant (ms)")
    w_min: float = Field(default=0.0, description="Minimum synaptic weight")
    w_max: float = Field(default=1.0, description="Maximum synaptic weight")


class NetworkConfig(BaseModel):
    """Configuration for the overall Spiking Neural Network."""

    model_config = ConfigDict(extra="forbid")

    layers: list[int] = Field(
        default=[32, 16, 8], description="Number of neurons in each layer"
    )
    neuron: LIFConfig = Field(default_factory=LIFConfig)
    plasticity: STDPConfig = Field(default_factory=STDPConfig)
    layer_gains: list[float] | None = Field(
        default=None,
        description=(
            "Per-layer synaptic gain, one entry per neuron group "
            "(len(layers) - 1). Overrides neuron.g_syn layer by layer. "
            "None keeps the single global neuron.g_syn everywhere."
        ),
    )

    @model_validator(mode="after")
    def _check_layer_gains(self) -> "NetworkConfig":
        """Per-layer gains must supply exactly one value per neuron group."""
        if self.layer_gains is not None:
            expected = len(self.layers) - 1
            if len(self.layer_gains) != expected:
                raise ValueError(
                    f"layer_gains has {len(self.layer_gains)} entries but "
                    f"{expected} neuron groups are defined by layers={self.layers}."
                )
            if any(g <= 0.0 for g in self.layer_gains):
                raise ValueError("layer_gains entries must be strictly positive.")
        return self

    def gain_for_layer(self, index: int) -> float:
        """Resolve the synaptic gain for neuron group `index`."""
        if self.layer_gains is None:
            return self.neuron.g_syn
        return self.layer_gains[index]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkConfig":
        """Create a config from a dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkConfig":
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "NetworkConfig":
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
