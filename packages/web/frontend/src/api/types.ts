export interface SpikeEvent {
  t_ms: number;
  layer: number;
  neuron: number;
}

export interface SimulateMetrics {
  input_snr_db: number;
  output_snr_db: number;
  delta_snr_db: number;
  rpeak_integrity: number;
  mean_firing_rate_hz: number;
}

export interface SimulateRequest {
  source: 'record' | 'upload';
  record_id?: string;
  signal_id?: string;
  duration_s: number;
  flicker_freq_hz: number;
  flicker_amplitude: number;
  input_snr_db: number;
  training_epochs: number;
  stdp_tau_ms: number;
  network_layers: number[];
  seed: number;
}

export interface SimulateResponse {
  simulation_id: string;
  fs: number;
  t_ms: number[];
  clean: number[];
  noisy: number[];
  reconstructed: number[];
  rpeaks_clean_ms: number[];
  rpeaks_reconstructed_ms: number[];
  spike_events: SpikeEvent[];
  metrics: SimulateMetrics;
  runtime_ms: number;
}

export interface RecordSummary {
  record_id: string;
  fs: number;
  duration_s: number;
  channels: string[];
}

export interface SegmentResponse {
  record_id: string;
  fs: number;
  signal: number[];
}

export interface UploadResponse {
  signal_id: string;
  fs: number;
  n_samples: number;
}
