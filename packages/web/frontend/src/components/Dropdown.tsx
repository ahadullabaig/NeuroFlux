import * as Select from '@radix-ui/react-select';
import { ChevronDown } from 'lucide-react';
import React from 'react';

interface DropdownProps {
  label: string;
  value: string;
  options: { value: string; label: string; sublabel?: string }[];
  onChange: (value: string) => void;
}

export const Dropdown: React.FC<DropdownProps> = ({ value, options, onChange }) => {
  return (
    <Select.Root value={value} onValueChange={onChange}>
      <Select.Trigger
        className="flex items-center justify-between w-full h-10 px-3 bg-surface-1 border border-rule-strong font-mono text-[12px] text-ink-1 hover:border-rule-hot transition-colors outline-none"
      >
        <Select.Value />
        <Select.Icon>
          <ChevronDown className="w-4 h-4 text-ink-3" />
        </Select.Icon>
      </Select.Trigger>

      <Select.Portal>
        <Select.Content className="overflow-hidden bg-surface-2 border border-rule-strong shadow-lg z-[100]">
          <Select.Viewport className="p-1">
            {options.map((opt) => (
              <Select.Item
                key={opt.value}
                value={opt.value}
                className="flex items-center justify-between px-3 py-2 font-mono text-[11px] text-ink-2 outline-none cursor-pointer hover:bg-surface-3 hover:text-accent-rust data-[selected]:text-accent-rust"
              >
                <Select.ItemText>{opt.label}</Select.ItemText>
                {opt.sublabel && (
                  <span className="ml-4 text-[9px] text-ink-4 opacity-70">{opt.sublabel}</span>
                )}
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
};
