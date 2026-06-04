import React, { useRef, useState, useEffect } from 'react';
import { xScale } from './plot/scales';
import type { SpikeEvent } from '../api/types';
import { motion } from 'framer-motion';

interface SpikeRasterProps {
  events: SpikeEvent[];
  durationS: number;
  layerSizes?: number[]; // e.g. [32, 16, 8]
}

export const SpikeRaster: React.FC<SpikeRasterProps> = ({
  events, durationS, layerSizes = [32, 16, 8]
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const height = 200;
  const margin = 40;
  const durationMs = durationS * 1000;

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      setWidth(entries[0].contentRect.width);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const totalNeurons = layerSizes.reduce((a, b) => a + b, 0);
  const layerOffsets = (function() {
    const offsets: number[] = [];
    let current = 0;
    for (let i = 0; i < layerSizes.length; i++) {
      offsets.push(current);
      current += layerSizes[i];
    }
    return offsets;
  })();

  const getY = (layer: number, neuron: number) => {
    const lIdx = layer - 1;
    const absoluteNeuronIdx = layerOffsets[lIdx] + neuron;
    const padding = 20;
    const availableHeight = height - 2 * padding;
    return height - padding - (absoluteNeuronIdx / (totalNeurons - 1)) * availableHeight;
  };

  const getLayerColor = (layer: number) => {
    switch (layer) {
      case 1: return "var(--color-accent-sage)";
      case 2: return "var(--color-accent-rust)";
      case 3: return "var(--color-accent-parchment)";
      default: return "var(--color-ink-3)";
    }
  };

  return (
    <div ref={containerRef} className="w-full h-[200px] bg-surface-1 border-t border-rule relative overflow-hidden">
      <svg width="100%" height="100%" className="block">
        {/* Layer Separators */}
        {layerOffsets.slice(1).map((_, i) => {
           const y = getY(i + 1, layerSizes[i]) + (getY(i + 2, 0) - getY(i + 1, layerSizes[i])) / 2;
           return (
             <line 
               key={i} 
               x1={margin} y1={y} x2={width - margin} y2={y} 
               stroke="var(--color-rule)" strokeWidth="1" strokeDasharray="2 4" 
             />
           );
        })}

        {/* Vertical Ticks matching Hero Plot */}
        <g className="stroke-rule/20">
          {[...Array(11)].map((_, i) => {
            const x = xScale(i * (durationMs / 10), durationMs, width, margin);
            return <line key={i} x1={x} y1={0} x2={x} y2={height} />;
          })}
        </g>

        {/* Spike Events */}
        {width > 0 && events.map((ev, i) => (
          <motion.rect
            key={i}
            x={xScale(ev.t_ms, durationMs, width, margin) - 1}
            y={getY(ev.layer, ev.neuron) - 2}
            width="2"
            height="4"
            fill={getLayerColor(ev.layer)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ 
              duration: 0.3, 
              delay: 0.6 + (ev.t_ms / durationMs) * 0.7 
            }}
          />
        ))}

        {/* Layer Labels */}
        <g className="font-mono text-[9px] uppercase fill-ink-4 select-none">
           {layerSizes.map((size, i) => (
             <text 
               key={i} 
               x={width - 5} 
               y={getY(i + 1, size / 2)} 
               textAnchor="end" 
               dominantBaseline="middle"
             >
               L{i+1} · {i === 0 ? 'IN' : i === 1 ? 'HID' : 'OUT'} · {size}
             </text>
           ))}
        </g>
      </svg>
    </div>
  );
};
