import React, { useMemo, useRef, useState, useEffect } from 'react';
import { buildPath, xScale, yScale } from './plot/scales';
import { motion, AnimatePresence } from 'framer-motion';

interface SignalPlotProps {
  clean: number[];
  noisy: number[];
  reconstructed: number[];
  rpeaksCleanMs: number[];
  rpeaksReconstructedMs: number[];
  durationS: number;
}

export const SignalPlot: React.FC<SignalPlotProps> = ({
  clean, noisy, reconstructed, rpeaksCleanMs, rpeaksReconstructedMs, durationS
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const margin = 40;
  
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const durationMs = durationS * 1000;
  
  const vMin = useMemo(() => {
    const all = [...clean, ...noisy, ...reconstructed];
    return all.length ? Math.min(...all) - 0.2 : -1;
  }, [clean, noisy, reconstructed]);
  
  const vMax = useMemo(() => {
    const all = [...clean, ...noisy, ...reconstructed];
    return all.length ? Math.max(...all) + 0.2 : 1;
  }, [clean, noisy, reconstructed]);

  const paths = useMemo(() => {
    if (dimensions.width === 0) return { clean: "", noisy: "", reconstructed: "" };
    return {
      clean: buildPath(clean, durationMs, dimensions.width, dimensions.height, margin, vMin, vMax),
      noisy: buildPath(noisy, durationMs, dimensions.width, dimensions.height, margin, vMin, vMax),
      reconstructed: buildPath(reconstructed, durationMs, dimensions.width, dimensions.height, margin, vMin, vMax),
    };
  }, [clean, noisy, reconstructed, dimensions, vMin, vMax, durationMs]);

  const [hoverT, setHoverT] = useState<number | null>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < margin || x > dimensions.width - margin) {
      setHoverT(null);
      return;
    }
    const t = ((x - margin) / (dimensions.width - 2 * margin)) * durationMs;
    setHoverT(t);
  };

  return (
    <div 
      ref={containerRef} 
      className="w-full h-full bg-surface-1 border border-rule relative group overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverT(null)}
    >
      {/* Scanline overlay */}
      <div className="absolute inset-0 pointer-events-none z-10 opacity-[0.015] bg-[repeating-linear-gradient(0deg,transparent,transparent_1px,rgba(255,255,255,0.5)_1px,rgba(255,255,255,0.5)_2px)]"></div>
      
      {/* Plot Caption */}
      <div className="absolute top-5 left-6 z-20 flex flex-col gap-0.5">
        <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-ink-3">Hero Trace · Time-Domain Reconstruction</span>
        <span className="font-display italic text-[20px] text-ink-1">Signal recovered through spiking dynamics</span>
      </div>

      {/* Legend */}
      <div className="absolute top-5 right-6 z-20 bg-surface-0/80 backdrop-blur-md border border-rule-strong p-3 flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <div className="w-4 h-[1.5px] bg-signal-noisy"></div>
          <span className="font-mono text-[9px] uppercase text-ink-2">Noisy Input</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-4 h-[1.5px] bg-signal-clean"></div>
          <span className="font-mono text-[9px] uppercase text-ink-2">Clean Ground Truth</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-4 h-[2px] bg-signal-reconstructed shadow-glow-rust"></div>
          <span className="font-mono text-[9px] uppercase text-ink-1">NeuroFlux Output</span>
        </div>
      </div>

      <svg width="100%" height="100%" className="block">
        {/* Grid Lines */}
        <g className="stroke-rule/30">
          {[...Array(11)].map((_, i) => {
            const x = xScale(i * (durationMs / 10), durationMs, dimensions.width, margin);
            return <line key={`v-${i}`} x1={x} y1={margin} x2={x} y2={dimensions.height - margin} />;
          })}
          {[...Array(5)].map((_, i) => {
            const y = yScale(vMin + i * ((vMax - vMin) / 4), vMin, vMax, dimensions.height, margin);
            return <line key={`h-${i}`} x1={margin} y1={y} x2={dimensions.width - margin} y2={y} />;
          })}
        </g>

        {/* Traces */}
        <AnimatePresence>
          {paths.noisy && (
            <motion.path
              key={`noisy-${noisy.length}`}
              d={paths.noisy}
              fill="none"
              stroke="var(--color-signal-noisy, #88A57B)"
              strokeWidth="1.1"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
            />
          )}
          {paths.clean && (
            <motion.path
              key={`clean-${clean.length}`}
              d={paths.clean}
              fill="none"
              stroke="var(--color-signal-clean, #B5AB95)"
              strokeWidth="1.3"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            />
          )}
          {paths.reconstructed && (
            <motion.path
              key={`recon-${reconstructed.length}`}
              d={paths.reconstructed}
              fill="none"
              stroke="var(--color-signal-reconstructed, #C04E2C)"
              strokeWidth="1.8"
              className="drop-shadow-[0_0_5px_rgba(192,78,44,0.35)]"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
            />
          )}
        </AnimatePresence>

        {/* R-Peaks */}
        {rpeaksCleanMs.map((t, i) => (
          <motion.path
            key={`clean-${i}`}
            d={`M ${xScale(t, durationMs, dimensions.width, margin)} ${yScale(clean[Math.round(t / (durationMs / (clean.length-1)))], vMin, vMax, dimensions.height, margin) - 10} l -4 8 l 8 0 z`}
            fill="none"
            stroke="var(--color-signal-clean)"
            strokeWidth="1"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.2 + i * 0.02 }}
          />
        ))}
        {rpeaksReconstructedMs.map((t, i) => (
          <motion.path
            key={`recon-${i}`}
            d={`M ${xScale(t, durationMs, dimensions.width, margin)} ${yScale(reconstructed[Math.round(t / (durationMs / (reconstructed.length-1)))], vMin, vMax, dimensions.height, margin) - 10} l -4 8 l 8 0 z`}
            fill="var(--color-accent-rust)"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.4 + i * 0.02 }}
          />
        ))}

        {/* Playhead */}
        {hoverT !== null && (
          <g>
            <line 
              x1={xScale(hoverT, durationMs, dimensions.width, margin)} 
              y1={margin} 
              x2={xScale(hoverT, durationMs, dimensions.width, margin)} 
              y2={dimensions.height - margin} 
              stroke="var(--color-rule-hot)" 
              strokeWidth="1" 
              strokeDasharray="4 4" 
            />
          </g>
        )}
      </svg>

      {/* Playhead Readout */}
      {hoverT !== null && (
        <div className="absolute bottom-5 right-6 z-20 bg-surface-0/60 backdrop-blur-lg border border-rule-strong p-2 font-mono text-[10px] flex flex-col gap-1 min-w-[120px]">
          <div className="flex justify-between border-b border-rule pb-1 mb-1">
            <span className="text-ink-4">TIME</span>
            <span className="text-ink-1">{(hoverT / 1000).toFixed(3)}s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-accent-sage">NOISY</span>
            <span className="text-ink-1">
              {noisy[Math.round(hoverT / (durationMs / (noisy.length - 1)))]?.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-accent-rust">RECON</span>
            <span className="text-ink-1">
              {reconstructed[Math.round(hoverT / (durationMs / (reconstructed.length - 1)))]?.toFixed(4)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
