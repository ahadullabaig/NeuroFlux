import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SimulateRequest, SimulateResponse, SimulateMetrics } from '../api/types';
import * as api from '../api/client';

export interface HistoryEntry {
  id: string;
  timestamp: number;
  params: SimulateRequest;
  metrics: SimulateMetrics;
}

interface SimulationState {
  params: SimulateRequest;
  setParams: (patch: Partial<SimulateRequest>) => void;
  status: 'idle' | 'running' | 'error';
  response: SimulateResponse | null;
  error: string | null;
  history: HistoryEntry[];
  
  runSimulation: () => Promise<void>;
  addToHistory: (entry: HistoryEntry) => void;
  clearError: () => void;
}

export const useSimulationStore = create<SimulationState>()(
  persist(
    (set, get) => ({
      params: {
        source: 'record',
        record_id: '214',
        duration_s: 10,
        flicker_freq_hz: 50,
        flicker_amplitude: 0.5,
        input_snr_db: 0,
        training_epochs: 10,
        stdp_tau_ms: 10,
        network_layers: [32, 16, 8],
        seed: 0,
      },
      status: 'idle',
      response: null,
      error: null,
      history: [],

      setParams: (patch) => set((state) => ({ 
        params: { ...state.params, ...patch } 
      })),

      clearError: () => set({ error: null }),

      addToHistory: (entry) => set((state) => ({
        history: [entry, ...state.history].slice(0, 20)
      })),

      runSimulation: async () => {
        set({ status: 'running', error: null });
        try {
          const response = await api.simulate(get().params);
          set({ response, status: 'idle' });
          
          get().addToHistory({
            id: response.simulation_id,
            timestamp: Date.now(),
            params: { ...get().params },
            metrics: response.metrics,
          });
        } catch (err: any) {
          set({ status: 'error', error: err.message || 'Simulation failed' });
        }
      },
    }),
    {
      name: 'neuroflux-simulation-storage',
      partialize: (state) => ({ history: state.history, params: state.params }),
    }
  )
);
