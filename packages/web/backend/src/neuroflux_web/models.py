import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List, Optional

class SpikeEvent(BaseModel):
    t_ms: float
    layer: int
    neuron: int

class SimulateMetrics(BaseModel):
    input_snr_db: float
    output_snr_db: float
    delta_snr_db: float
    rpeak_integrity: float
    mean_firing_rate_hz: float

class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["record", "upload"]
    record_id: Optional[str] = None                # required if source=="record"
    signal_id: Optional[str] = None                # required if source=="upload"
    duration_s: float = Field(default=10.0, gt=0, le=30.0)
    flicker_freq_hz: float = Field(default=50.0, ge=0.0, le=200.0)
    flicker_amplitude: float = Field(default=0.5, ge=0.0, le=2.0)
    input_snr_db: float = Field(default=0.0, ge=-10.0, le=30.0)
    training_epochs: int = Field(default=10, ge=0, le=100)
    stdp_tau_ms: float = Field(default=10.0, gt=0, le=100.0)
    network_layers: List[int] = Field(default_factory=lambda: [32, 16, 8])
    seed: int = 0

class SimulateResponse(BaseModel):
    simulation_id: str                          # uuid4 hex
    fs: float
    t_ms: List[float]                           # length N (subsample if > 5000)
    clean: List[float]
    noisy: List[float]
    reconstructed: List[float]
    rpeaks_clean_ms: List[float]
    rpeaks_reconstructed_ms: List[float]
    spike_events: List[SpikeEvent]              # sparse: (t_ms, layer, neuron)
    metrics: SimulateMetrics
    runtime_ms: float
