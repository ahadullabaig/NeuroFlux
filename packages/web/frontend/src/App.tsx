import { Layout } from './components/Layout';
import { useSimulationStore } from './store/simulationStore';
import { ControlsPanel } from './components/ControlsPanel';
import { SignalPlot } from './components/SignalPlot';
import { SpikeRaster } from './components/SpikeRaster';
import { MetricsPanel } from './components/MetricsPanel';
import { HistoryPanel } from './components/HistoryPanel';
import { AlertCircle } from 'lucide-react';

function App() {
  const { status, response, error, clearError, params } = useSimulationStore();

  const sidebar = <ControlsPanel />;

  const main = (
    <div className="flex flex-col gap-6 h-full">
      {/* Error Banner */}
      {error && (
        <div className="bg-accent-rust/10 border border-accent-rust/30 p-4 flex items-center justify-between">
          <div className="flex items-center gap-3 text-accent-rust">
            <AlertCircle className="w-5 h-5" />
            <span className="font-mono text-[12px] uppercase">{error}</span>
          </div>
          <button onClick={clearError} className="text-ink-3 hover:text-ink-1 font-mono text-[11px] uppercase underline underline-offset-4">Dismiss</button>
        </div>
      )}

      {/* Hero Canvas */}
      <div className="flex-1 min-h-0 flex flex-col border border-rule bg-surface-1">
        {response ? (
          <>
            <div className="flex-1 min-h-0">
              <SignalPlot 
                clean={response.clean}
                noisy={response.noisy}
                reconstructed={response.reconstructed}
                rpeaksCleanMs={response.rpeaks_clean_ms}
                rpeaksReconstructedMs={response.rpeaks_reconstructed_ms}
                durationS={params.duration_s}
              />
            </div>
            <SpikeRaster 
              events={response.spike_events}
              durationS={params.duration_s}
              layerSizes={params.network_layers}
            />
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center relative overflow-hidden group">
            {/* Background scanline for empty state too */}
            <div className="absolute inset-0 pointer-events-none opacity-[0.015] bg-[repeating-linear-gradient(0deg,transparent,transparent_1px,rgba(255,255,255,0.5)_1px,rgba(255,255,255,0.5)_2px)]"></div>
            
            <div className="flex flex-col items-center gap-4 z-10">
              <div className={`w-3 h-3 rounded-full bg-accent-rust ${status === 'running' ? 'animate-ping' : 'opacity-20'}`}></div>
              <div className="flex flex-col items-center gap-1">
                <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-2">
                  {status === 'running' ? 'Simulating Dynamic Response...' : 'Awaiting Instrument Activation'}
                </span>
                <span className="font-ui text-[11px] text-ink-4 italic">Configure parameters and initiate run via the left panel</span>
              </div>
            </div>

            {/* Subtle tech border accents */}
            <div className="absolute top-4 left-4 w-4 h-4 border-l border-t border-rule opacity-50"></div>
            <div className="absolute top-4 right-4 w-4 h-4 border-r border-t border-rule opacity-50"></div>
            <div className="absolute bottom-4 left-4 w-4 h-4 border-l border-b border-rule opacity-50"></div>
            <div className="absolute bottom-4 right-4 w-4 h-4 border-r border-b border-rule opacity-50"></div>
          </div>
        )}
      </div>

      {/* Metrics Row */}
      <MetricsPanel 
        metrics={response?.metrics ?? null} 
        runtimeMs={response?.runtime_ms} 
        simulationId={response?.simulation_id}
      />

      {/* History Strip */}
      <HistoryPanel />

      {/* Bottom status bar */}
      <footer className="flex justify-between items-center py-2 px-1 font-mono text-[9px] text-ink-4 uppercase tracking-[0.1em]">
        <div className="flex gap-4">
          <span>Engine: Numba/JIT v0.58</span>
          <span>Buffer: 250Hz Circular</span>
        </div>
        <div>
          {response?.simulation_id && (
            <span className="text-ink-3">UUID: {response.simulation_id}</span>
          )}
        </div>
      </footer>
    </div>
  );

  return (
    <Layout sidebar={sidebar} main={main} />
  );
}

export default App;
