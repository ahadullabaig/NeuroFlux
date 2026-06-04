import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { uploadCsv } from '../api/client';
import { useSimulationStore } from '../store/simulationStore';

export const UploadDropzone: React.FC = () => {
  const { setParams } = useSimulationStore();
  const [status, setStatus] = React.useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [error, setError] = React.useState<string | null>(null);
  const [fileInfo, setFileInfo] = React.useState<{ name: string; n_samples: number } | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setStatus('uploading');
    setError(null);

    try {
      const res = await uploadCsv(file, 250);
      setParams({ signal_id: res.signal_id, source: 'upload', record_id: undefined });
      setFileInfo({ name: file.name, n_samples: res.n_samples });
      setStatus('success');
    } catch (err: any) {
      setStatus('error');
      setError(err.message || 'Upload failed');
    }
  }, [setParams]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  });

  return (
    <div className="flex flex-col gap-3">
      <div 
        {...getRootProps()} 
        className={`
          border-2 border-dashed p-6 flex flex-col items-center gap-3 cursor-pointer transition-all
          ${isDragActive ? 'border-accent-rust bg-accent-rust/5' : 'border-rule hover:border-rule-hot bg-surface-1'}
          ${status === 'success' ? 'border-accent-sage/50 bg-accent-sage/5' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {status === 'uploading' ? (
          <Loader2 className="w-8 h-8 text-ink-3 animate-spin" />
        ) : status === 'success' ? (
          <CheckCircle2 className="w-8 h-8 text-accent-sage" />
        ) : (
          <Upload className={`w-8 h-8 ${isDragActive ? 'text-accent-rust' : 'text-ink-4'}`} />
        )}

        <div className="flex flex-col items-center gap-1 text-center">
          <span className="font-mono text-[11px] uppercase tracking-wider text-ink-1">
            {status === 'uploading' ? 'Analyzing Signal...' : 
             status === 'success' ? 'Ready to Simulate' :
             isDragActive ? 'Drop CSV here' : 'Drop CSV or click to upload'}
          </span>
          <span className="font-ui text-[10px] text-ink-4">
            {status === 'success' ? fileInfo?.name : 'Single-column ECG @ 250Hz'}
          </span>
        </div>
      </div>

      {status === 'error' && (
        <div className="flex items-center gap-2 text-accent-mustard bg-accent-mustard/10 p-2 border border-accent-mustard/20">
          <XCircle className="w-4 h-4" />
          <span className="font-mono text-[9px] uppercase">{error}</span>
        </div>
      )}

      {status === 'success' && (
        <div className="flex flex-col gap-1 p-2 bg-surface-2 border border-rule">
           <div className="flex justify-between items-center">
             <span className="font-mono text-[9px] uppercase text-ink-3">Samples</span>
             <span className="font-mono text-[10px] text-ink-1 tabular-nums">{fileInfo?.n_samples}</span>
           </div>
           <div className="flex justify-between items-center">
             <span className="font-mono text-[9px] uppercase text-ink-3">Est. Duration</span>
             <span className="font-mono text-[10px] text-ink-1 tabular-nums">{(fileInfo!.n_samples / 250).toFixed(1)}s</span>
           </div>
        </div>
      )}
    </div>
  );
};
