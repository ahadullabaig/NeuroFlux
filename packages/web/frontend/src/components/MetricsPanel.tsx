import React from 'react';
import type { SimulateMetrics } from '../api/types';
import { motion } from 'framer-motion';
import { Download } from 'lucide-react';
import { exportCsvUrl } from '../api/client';

interface MetricsPanelProps {
  metrics: SimulateMetrics | null;
  runtimeMs?: number;
  simulationId?: string;
}

const MetricCell: React.FC<{ 
  label: string; 
  value: string | number; 
  unit?: string; 
  colorClass?: string;
  subline?: React.ReactNode;
}> = ({ label, value, unit, colorClass = "text-ink-1", subline }) => (
  <div className="bg-surface-1 p-5 flex flex-col gap-2 relative overflow-hidden">
    <span className="font-mono text-[9px] uppercase tracking-widest text-ink-3">{label}</span>
    <div className="flex items-baseline gap-1">
      <span className={`font-mono text-[28px] tabular-nums ${colorClass}`}>
        {typeof value === 'number' ? value.toFixed(3) : value}
      </span>
      {unit && <span className="font-mono text-[10px] text-ink-3 uppercase">{unit}</span>}
    </div>
    {subline && <div className="mt-1">{subline}</div>}
  </div>
);

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics, runtimeMs, simulationId }) => {
  if (!metrics) {
    return (
      <div className="grid grid-cols-5 gap-px bg-rule border border-rule">
        {['ΔSNR', 'RPEAK', 'SNR IN', 'FIRE RATE', 'RUNTIME'].map((l) => (
          <MetricCell key={l} label={l} value="—" />
        ))}
      </div>
    );
  }

  const deltaColor = metrics.delta_snr_db >= 0 ? "text-accent-rust" : "text-accent-mustard";
  const rpeakColor = metrics.rpeak_integrity >= 0.85 ? "bg-accent-rust" : "bg-accent-mustard";

  return (
    <div className="grid grid-cols-5 gap-px bg-rule border border-rule">
      <MetricCell 
        label="ΔSNR" 
        value={metrics.delta_snr_db} 
        unit="dB" 
        colorClass={deltaColor}
        subline={<div className={`h-[3px] w-full ${metrics.delta_snr_db >= 0 ? 'bg-accent-rust/20' : 'bg-accent-mustard/20'}`}></div>}
      />
      <MetricCell 
        label="RPEAK Integrity" 
        value={metrics.rpeak_integrity} 
        subline={
          <div className="h-[3px] w-full bg-surface-3">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${metrics.rpeak_integrity * 100}%` }}
              className={`h-full ${rpeakColor}`}
            />
          </div>
        }
      />
      <MetricCell label="Input SNR" value={metrics.input_snr_db} unit="dB" />
      <MetricCell label="Mean Fire" value={metrics.mean_firing_rate_hz} unit="Hz" />
      <MetricCell label="Runtime" value={runtimeMs ?? 0} unit="ms" />
      
      <div className="bg-surface-1 p-5 flex flex-col justify-center items-center gap-2 relative overflow-hidden group">
         {simulationId ? (
           <a 
             href={exportCsvUrl(simulationId)} 
             download 
             className="w-full h-full flex flex-col items-center justify-center gap-2 text-ink-3 hover:text-accent-rust transition-colors"
           >
             <Download className="w-6 h-6" />
             <span className="font-mono text-[9px] uppercase tracking-widest text-center">Export CSV</span>
           </a>
         ) : (
            <div className="flex flex-col items-center gap-2 opacity-20">
              <Download className="w-6 h-6" />
              <span className="font-mono text-[9px] uppercase tracking-widest">Export CSV</span>
            </div>
         )}
      </div>
    </div>
  );
};
