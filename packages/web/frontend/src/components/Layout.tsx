import React from 'react';

interface LayoutProps {
  sidebar: React.ReactNode;
  main: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ sidebar, main }) => {
  return (
    <div className="grid grid-cols-[320px_1fr] grid-rows-[56px_1fr] h-screen w-screen overflow-hidden relative">
      {/* Background grain/noise overlay - removed external URL */}
      <div className="pointer-events-none fixed inset-0 z-50 opacity-[0.01] mix-blend-overlay bg-black"></div>
      
      {/* Header */}
      <header className="col-span-2 border-b border-rule flex items-center justify-between px-6 bg-surface-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 border border-accent-rust flex items-center justify-center">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-rust">
              <path d="M2 12C2 12 5 2 12 2C19 2 22 12 22 12C22 12 19 22 12 22C5 22 2 12 2 12Z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <span className="font-ui font-semibold tracking-[0.18em] uppercase text-[13px]">
            NEURO<span className="font-display italic text-[17px] text-accent-rust normal-case tracking-normal">flux</span>
          </span>
        </div>
        
        {/* Meta Readout */}
        <div className="font-mono text-[11px] uppercase flex items-center gap-2 text-ink-3">
          <span>K v</span> <span className="text-ink-4">·</span>
          <span>K v</span> <span className="text-ink-4">·</span>
          <span>K v</span>
          <span className="w-1 h-3 bg-accent-rust animate-pulse ml-1"></span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            <div className="flex items-center gap-1.5 border border-rule px-2 py-0.5 rounded-sm bg-surface-1">
              <div className="w-1.5 h-1.5 rounded-full bg-accent-sage shadow-[0_0_6px_rgba(136,165,123,0.5)]"></div>
              <span className="font-mono text-[9px] text-ink-2 uppercase">NUMBA WARM</span>
            </div>
          </div>
          <button className="text-[10px] font-mono text-ink-3 hover:text-ink-1 uppercase tracking-wider transition-colors">Take the tour</button>
        </div>
      </header>

      {/* Sidebar / Controls */}
      <aside className="border-r border-rule bg-surface-0 flex flex-col overflow-y-auto">
        {sidebar}
      </aside>

      {/* Main Canvas */}
      <main className="bg-void p-6 flex flex-col gap-6 overflow-y-auto">
        {main}
      </main>
    </div>
  );
};
