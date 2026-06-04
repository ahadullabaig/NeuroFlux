import time
import numpy as np
import uuid
from typing import List, Tuple
from neuroflux_web.models import SimulateRequest, SimulateResponse, SpikeEvent, SimulateMetrics
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_core.network.simulator import Simulator
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.io.config import LIFConfig, STDPConfig
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_core.metrics.rpeak import rpeak_integrity
from scipy.signal import find_peaks
from neuroflux_datasets.synthetic import add_flicker, add_gaussian_noise

def _find_rpeaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Local helper matching neuroflux_core.metrics.rpeak logic."""
    std = float(np.std(signal))
    prom = max(std * 0.5, 1e-10)
    peaks, _ = find_peaks(signal, prominence=prom)
    return peaks

def _reset_transient_state(sim: Simulator):
    sim.time = 0.0
    sim.encoder.reset_state()
    sim.decoder.reset_state()
    net = sim.network
    for n in net.neurons:
        n.v.fill(n.config.v_rest)
        n.refractory_time.fill(0.0)
    for s in net.synapses:
        s.trace_pre.fill(0.0)
        s.trace_post.fill(0.0)

def run_simulation(req: SimulateRequest, clean_signal: np.ndarray, fs: float) -> SimulateResponse:
    start_time = time.time()
    
    # 1. Inject Flicker and Noise
    rng = np.random.default_rng(req.seed)
    noisy_signal = clean_signal.copy()
    if req.flicker_freq_hz > 0 and req.flicker_amplitude > 0:
        noisy_signal = add_flicker(noisy_signal, fs, req.flicker_freq_hz, req.flicker_amplitude)
    
    noisy_signal = add_gaussian_noise(noisy_signal, req.input_snr_db, rng)
    
    # 2. Setup SNN
    dt = 1000.0 / fs
    n_input = req.network_layers[0]
    
    net = FeedForwardNetwork.build(
        layers=req.network_layers,
        neuron=LIFConfig(),
        plasticity=STDPConfig(tau_plus=req.stdp_tau_ms, tau_minus=req.stdp_tau_ms),
        seed=req.seed
    )
    
    v_min = float(np.min(noisy_signal))
    v_max = float(np.max(noisy_signal))
    if v_max <= v_min:
        v_max = v_min + 1.0
        
    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=100.0,
        rng=np.random.default_rng(req.seed)
    )
    decoder = PopulationDecoder(v_min=v_min, v_max=v_max, max_firing_rate_hz=100.0)
    
    sim = Simulator(net, dt=dt, seed=req.seed, encoder=encoder, decoder=decoder)
    
    # Tile signal across input layer
    time_varying_noisy = np.tile(noisy_signal, (n_input, 1)).T
    duration_ms = req.duration_s * 1000.0
    
    # 3. Training
    if req.training_epochs > 0:
        for _ in range(req.training_epochs):
            sim.run(time_varying_noisy, duration_ms=duration_ms, training=True)
            _reset_transient_state(sim)
            
    # 4. Final Eval + Spike Collection
    n_steps = int(round(duration_ms / dt))
    reconstructed = np.zeros(n_steps)
    spike_events: List[SpikeEvent] = []
    
    total_non_input_spikes = 0
    total_non_input_neurons = sum(req.network_layers[1:])
    
    for t in range(n_steps):
        frame = time_varying_noisy[t]
        decoded, layer_spikes = sim.run_step(frame, training=False)
        # decoded is often a 1-element array from PopulationDecoder
        reconstructed[t] = float(np.asarray(decoded).item())
        
        # Collect spikes (sparse)
        for layer_idx, spikes in enumerate(layer_spikes):
            is_input = (layer_idx == 0)
            if not is_input:
                total_non_input_spikes += np.sum(spikes)
            
            for neuron_idx in np.where(spikes)[0]:
                spike_events.append(SpikeEvent(
                    t_ms=float(t * dt),
                    layer=layer_idx + 1,
                    neuron=int(neuron_idx)
                ))
    
    # Cap spike events at 5000
    if len(spike_events) > 5000:
        indices = np.linspace(0, len(spike_events) - 1, 5000, dtype=int)
        spike_events = [spike_events[i] for i in indices]
        
    # 5. Metrics
    out_snr = calculate_snr(clean_signal, reconstructed)
    in_snr = calculate_snr(clean_signal, noisy_signal)
    rpeak_recon = rpeak_integrity(clean_signal, reconstructed, fs)
    mean_fr = total_non_input_spikes / (req.duration_s * total_non_input_neurons) if total_non_input_neurons > 0 else 0.0
    
    # 6. R-peaks locations
    peaks_clean = _find_rpeaks(clean_signal, fs)
    peaks_recon = _find_rpeaks(reconstructed, fs)
    
    rpeaks_clean_ms = (peaks_clean * (1000.0 / fs)).tolist()
    rpeaks_reconstructed_ms = (peaks_recon * (1000.0 / fs)).tolist()
    
    # Subsample t_ms/signals if > 5000 points
    t_ms = (np.arange(n_steps) * dt).tolist()
    clean_list = clean_signal.tolist()
    noisy_list = noisy_signal.tolist()
    recon_list = reconstructed.tolist()
    
    if n_steps > 5000:
        indices = np.linspace(0, n_steps - 1, 5000, dtype=int)
        t_ms = [float(t_ms[i]) for i in indices]
        clean_list = [float(clean_list[i]) for i in indices]
        noisy_list = [float(noisy_list[i]) for i in indices]
        recon_list = [float(recon_list[i]) for i in indices]

    metrics = SimulateMetrics(
        input_snr_db=float(in_snr),
        output_snr_db=float(out_snr),
        delta_snr_db=float(out_snr - in_snr),
        rpeak_integrity=float(rpeak_recon),
        mean_firing_rate_hz=float(mean_fr)
    )
    
    return SimulateResponse(
        simulation_id=uuid.uuid4().hex,
        fs=fs,
        t_ms=t_ms,
        clean=clean_list,
        noisy=noisy_list,
        reconstructed=recon_list,
        rpeaks_clean_ms=rpeaks_clean_ms,
        rpeaks_reconstructed_ms=rpeaks_reconstructed_ms,
        spike_events=spike_events,
        metrics=metrics,
        runtime_ms=(time.time() - start_time) * 1000.0
    )
