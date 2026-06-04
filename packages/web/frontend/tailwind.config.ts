import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void:    '#0F1B2E',
        surface: {
          0: '#122036',
          1: '#15243C',
          2: '#1A2C46',
          3: '#213658',
        },
        rule: {
          DEFAULT: '#2B3C5A',
          strong:  '#3D5378',
          hot:     '#557099',
        },
        ink: {                          // "text" namespace — parchment scale
          1: '#EDE4D3',
          2: '#B5AB95',
          3: '#7E7660',
          4: '#4A5670',
        },
        signal: {
          clean:         '#B5AB95',
          noisy:         '#88A57B',
          reconstructed: '#C04E2C',
        },
        accent: {
          rust:      '#C04E2C',         // brand · L2 · positive Δ
          sage:      '#88A57B',         // noisy · L1
          mustard:   '#D4C158',         // negative Δ
          parchment: '#EDE4D3',         // L3 · primary ink
        },
      },
      fontFamily: {
        display: ['"Instrument Serif"', 'serif'],
        ui:      ['"Inter Tight"', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', 'monospace'],
      },
      transitionTimingFunction: {
        'instrument-out':   'cubic-bezier(.22, 1, .36, 1)',
        'instrument-inout': 'cubic-bezier(.65, 0, .35, 1)',
      },
      boxShadow: {
        'glow-rust':    '0 0 8px rgba(192,78,44,0.4)',
        'glow-mustard': '0 0 6px rgba(212,193,88,0.35)',
      },
    },
  },
  plugins: [],
} satisfies Config
