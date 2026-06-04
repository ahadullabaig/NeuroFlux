import React from 'react';
import { useSimulationStore, type HistoryEntry } from '../store/simulationStore';
import { Clock, Activity } from 'lucide-react';

export const HistoryPanel: React.FC = () => {
  const { history, setParams } = useSimulationStore();

  if (history.length === 0) return null;

  return (
    <div className="flex flex-col gap-px bg-rule border border-rule">
      <div className="bg-surface-1 px-5 py-3 flex justify-between items-center">
        <span className="font-mono text-[10px] uppercase tracking-widest text-ink-3">Session History</span>
        <span className="font-mono text-[11px] text-accent-rust">{history.length} runs</span>
      </div>
      
      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-px bg-rule">
        {history.map((entry) => (
          <HistoryItem key={entry.id} entry={entry} onRevisit={() => setParams(entry.params)} />
        ))}
      </div>
    </div>
  );
};

const HistoryItem: React.FC<{ entry: HistoryEntry; onRevisit: () => void }> = ({ entry, onRevisit }) => {
  const timeStr = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const deltaColor = entry.metrics.delta_snr_db >= 0 ? "text-accent-rust" : "text-accent-mustard";
  
  return (
    <div 
      className="bg-surface-0 hover:bg-surface-2 p-4 flex flex-col gap-3 transition-colors cursor-pointer group"
      onClick={onRevisit}
    >
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-2 text-ink-3">
          <Clock className="w-3 h-3" />
          <span className="font-mono text-[9px] tabular-nums">{timeStr}</span>
        </div>
        <div className={`font-mono text-[12px] font-bold ${deltaColor}`}>
          {entry.metrics.delta_snr_db >= 0 ? '+' : ''}{entry.metrics.delta_snr_db.toFixed(2)} dB
        </div>
      </div>
      
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <Activity className="w-3 h-3 text-ink-4" />
          <span className="font-mono text-[10px] text-ink-2 uppercase truncate">
            {entry.params.source === 'record' ? `Record ${entry.params.record_id}` : 'Custom Upload'}
          </span>
        </div>
        <div className="font-mono text-[9px] text-ink-4 flex gap-2">
          <span>{entry.params.input_snr_db}dB IN</span>
          <span>·</span>
          <span>{entry.params.training_epochs} EP</span>
          <span>·</span>
          <span>R{entry.metrics.rpeak_integrity.toFixed(3)}</span>
        </div>
      </div>
      
      <div className="opacity-0 group-hover:opacity-100 transition-opacity flex justify-end">
        <span className="font-mono text-[8px] uppercase tracking-tighter text-accent-rust">Click to Revisit Params</span>
      </div>
    </div>
  );
};
