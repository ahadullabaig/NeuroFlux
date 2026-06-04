import * as RadixSlider from '@radix-ui/react-slider';
import React from 'react';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  hint?: string;
  onChange: (value: number) => void;
}

export const Slider: React.FC<SliderProps> = ({ 
  label, value, min, max, step = 1, unit = '', hint, onChange 
}) => {
  return (
    <div className="flex flex-col gap-2 group">
      <div className="flex justify-between items-baseline">
        <div className="flex items-center gap-1.5">
          <span className="font-ui text-[12px] text-ink-1 font-medium">{label}</span>
          {hint && (
            <span className="font-mono text-[9px] text-ink-4 lowercase tabular-nums">{hint}</span>
          )}
        </div>
        <div className="font-mono text-[11px] text-ink-2 tabular-nums">
          {value}{unit}
        </div>
      </div>
      
      <RadixSlider.Root
        className="relative flex items-center select-none touch-none w-full h-5"
        value={[value]}
        max={max}
        min={min}
        step={step}
        onValueChange={([v]) => onChange(v)}
      >
        <RadixSlider.Track className="bg-surface-3 relative grow h-[2px]">
          <RadixSlider.Range className="absolute bg-accent-rust h-full shadow-[0_0_8px_rgba(192,78,44,0.4)]" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          className="block w-3 h-3 bg-accent-rust rotate-45 focus:outline-none transition-transform hover:scale-110 active:scale-95"
          aria-label={label}
        />
      </RadixSlider.Root>
      
      <div className="flex justify-between px-0.5">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="w-[1px] h-[3px] bg-ink-4"></div>
        ))}
      </div>
    </div>
  );
};
