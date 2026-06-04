import React, { useEffect, useState } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { Slider } from './Slider';
import { Dropdown } from './Dropdown';
import { listRecords } from '../api/client';
import type { RecordSummary } from '../api/types';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { UploadDropzone } from './UploadDropzone';
import { Play, Loader2, Database, Upload } from 'lucide-react';

export const ControlsPanel: React.FC = () => {
  const { params, setParams, runSimulation, status } = useSimulationStore();
  const [records, setRecords] = useState<RecordSummary[]>([]);
  const [isPassive, setIsPassive] = useState(params.training_epochs === 0);
  const [savedEpochs, setSavedEpochs] = useState(params.training_epochs || 10);

  useEffect(() => {
    listRecords().then(setRecords).catch(console.error);
  }, []);

  const handleModeChange = (val: string) => {
    if (!val) return;
    const passive = val === 'passive';
    setIsPassive(passive);
    if (passive) {
      setSavedEpochs(params.training_epochs);
      setParams({ training_epochs: 0 });
    } else {
      setParams({ training_epochs: savedEpochs || 10 });
    }
  };

  const recordOptions = records.map(r => ({
    value: r.record_id,
    label: `RECORD ${r.record_id}`,
    sublabel: `MIT-BIH · 30 MIN`
  }));

  return (
    <div className="p-5 flex flex-col gap-8 h-full">
      {/* 01 SOURCE */}
      <section>
        <div className="flex justify-between items-center mb-4 border-b border-rule pb-2">
          <h2 className="font-mono text-[11px] uppercase tracking-widest text-ink-2">01 Source</h2>
          <span className="font-mono text-[10px] text-ink-4">01</span>
        </div>
        
        <div className="flex flex-col gap-4">
          <ToggleGroup.Root
            type="single"
            className="flex bg-surface-1 border border-rule-strong p-0.5 rounded-sm mb-2"
            value={params.source}
            onValueChange={(val) => val && setParams({ source: val as any })}
          >
            <ToggleGroup.Item value="record" className="flex-1 py-1.5 flex items-center justify-center gap-2 font-mono text-[10px] uppercase outline-none data-[state=on]:bg-surface-3 data-[state=on]:text-accent-rust transition-colors">
              <Database className="w-3 h-3" />
              <span>Records</span>
            </ToggleGroup.Item>
            <ToggleGroup.Item value="upload" className="flex-1 py-1.5 flex items-center justify-center gap-2 font-mono text-[10px] uppercase outline-none data-[state=on]:bg-surface-3 data-[state=on]:text-accent-rust transition-colors">
              <Upload className="w-3 h-3" />
              <span>Upload</span>
            </ToggleGroup.Item>
          </ToggleGroup.Root>

          {params.source === 'record' ? (
            <Dropdown 
              label="Record"
              value={params.record_id || ''}
              options={recordOptions}
              onChange={(val) => setParams({ record_id: val, source: 'record' })}
            />
          ) : (
            <UploadDropzone />
          )}
        </div>
      </section>

      {/* 02 NOISE INJECTION */}
      <section>
        <div className="flex justify-between items-center mb-4 border-b border-rule pb-2">
          <h2 className="font-mono text-[11px] uppercase tracking-widest text-ink-2">02 Noise Injection</h2>
          <span className="font-mono text-[10px] text-ink-4">02</span>
        </div>
        <div className="flex flex-col gap-5">
          <Slider 
            label="Flicker Freq" 
            value={params.flicker_freq_hz} 
            min={0} max={200} unit="Hz" hint="Powerline interference"
            onChange={(v) => setParams({ flicker_freq_hz: v })}
          />
          <Slider 
            label="Flicker Amp" 
            value={params.flicker_amplitude} 
            min={0} max={2} step={0.1} hint="Relative strength"
            onChange={(v) => setParams({ flicker_amplitude: v })}
          />
          <Slider 
            label="Input SNR" 
            value={params.input_snr_db} 
            min={-10} max={30} unit="dB" hint="Signal degradation"
            onChange={(v) => setParams({ input_snr_db: v })}
          />
        </div>
      </section>

      {/* 03 STDP LEARNING */}
      <section>
        <div className="flex justify-between items-center mb-4 border-b border-rule pb-2">
          <h2 className="font-mono text-[11px] uppercase tracking-widest text-ink-2">03 STDP Learning</h2>
          <span className="font-mono text-[10px] text-ink-4">03</span>
        </div>
        <div className="flex flex-col gap-5">
          <Slider 
            label="Training Epochs" 
            value={isPassive ? 0 : (params.training_epochs || savedEpochs)} 
            min={0} max={100} hint="Unsupervised iterations"
            onChange={(v) => {
              if (!isPassive) setParams({ training_epochs: v });
              setSavedEpochs(v);
            }}
          />
          <Slider 
            label="STDP τ" 
            value={params.stdp_tau_ms} 
            min={1} max={100} unit="ms" hint="Trace decay constant"
            onChange={(v) => setParams({ stdp_tau_ms: v })}
          />
        </div>
      </section>

      {/* 04 MODE */}
      <section>
        <div className="flex justify-between items-center mb-4 border-b border-rule pb-2">
          <h2 className="font-mono text-[11px] uppercase tracking-widest text-ink-2">04 Mode</h2>
          <span className="font-mono text-[10px] text-ink-4">04</span>
        </div>
        
        <ToggleGroup.Root
          type="single"
          className="flex bg-surface-1 border border-rule-strong p-0.5 rounded-sm"
          value={isPassive ? 'passive' : 'trained'}
          onValueChange={handleModeChange}
        >
          <ToggleGroup.Item
            value="passive"
            className="flex-1 py-2 px-1 flex flex-col items-center gap-0.5 outline-none data-[state=on]:bg-surface-3 transition-colors group"
          >
            <span className="font-mono text-[11px] uppercase text-ink-2 group-data-[state=on]:text-accent-rust">Passive</span>
            <span className="font-mono text-[8px] text-ink-4 uppercase">0 ep · STDP OFF</span>
          </ToggleGroup.Item>
          <ToggleGroup.Item
            value="trained"
            className="flex-1 py-2 px-1 flex flex-col items-center gap-0.5 outline-none data-[state=on]:bg-surface-3 transition-colors group"
          >
            <span className="font-mono text-[11px] uppercase text-ink-2 group-data-[state=on]:text-accent-rust">Trained</span>
            <span className="font-mono text-[8px] text-ink-4 uppercase">{savedEpochs} ep · STDP ON</span>
          </ToggleGroup.Item>
        </ToggleGroup.Root>
        
        <p className="mt-3 text-[10px] font-ui text-ink-3 leading-relaxed italic border-l border-rule pl-3">
          {isPassive 
            ? "Regime baseline: temporal fidelity preserved exactly (rpeak = 1.000)."
            : "STDP optimization: amplitude noise reduction (+ΔSNR) via weight adaptation."}
        </p>
      </section>

      {/* RUN BUTTON */}
      <div className="mt-auto">
        <button 
          onClick={() => runSimulation()}
          disabled={status === 'running' || (params.source === 'record' && !params.record_id) || (params.source === 'upload' && !params.signal_id)}
          className="w-full h-14 bg-surface-2 border border-rule-hot font-mono text-[12px] uppercase tracking-[0.2em] text-ink-1 hover:border-accent-rust hover:text-accent-rust hover:shadow-glow-rust transition-all disabled:opacity-30 disabled:cursor-not-allowed group flex items-center justify-center gap-3"
        >
          {status === 'running' ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Running...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current group-hover:scale-110 transition-transform" />
              <span>Run Simulation</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
