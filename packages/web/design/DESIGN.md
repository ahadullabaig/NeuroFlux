# NeuroFlux — Phase 5 Design System

Visual + interaction direction for the Phase 5 dashboard. Read this **alongside
[`mockup.html`](./mockup.html)** — the mockup is the source of truth for
"what it should look like"; this doc is the source of truth for "why those
choices, and how to encode them in code."

> Open `mockup.html` in any modern browser. No build step. The signal data is
> procedurally generated, so the chart, raster, sliders, mode toggle, and
> metrics are all live and interactive.

---

## 1. Aesthetic direction — "blueprint scientific instrument"

Committed direction: a deep **prussian blueprint canvas** with parchment ink,
treated as a **technical drawing** rather than a SaaS dashboard or generic
dark UI. The blueprint metaphor is load-bearing — every choice (palette,
hairlines, mono numerics, typography) supports the read of "this is a drawn
instrument schematic, brought to life."

References pulled from (informally):

- 19th–20th century engineering blueprints — prussian blue paper, cream ink,
  precision drafting lines, marginal annotations in rust-orange grease pencil.
- Bloomberg Terminal's information density and mono-numeric data treatment.
- Editorial science journals (NEJM, Nature) for serif accent typography
  used **sparingly** in headers.
- The kind of marginalia annotations you'd see on an oscilloscope schematic
  — rust ink for emphasis, sage for "this is the noise floor."

**Why blueprint, not dark+cyan:** dark surfaces with cyan accents have become
the default AI-generated dashboard aesthetic in 2024–2025 (every Vercel
template, every shadcn theme). The blueprint reads as warmer, more
hand-drawn, more academically credible — and crucially, *uncommon* in
software UIs. Evaluators will not have seen it before.

The whole dashboard is one continuous instrument surface — header, controls,
canvas, metrics, history — separated by **hairline rules**, not cards with
heavy shadows. There is one container border, then everything inside is
demarcated by 1px rules in pale cream over the prussian. This is the single
biggest non-SaaS move.

**The rule:** no decoration that isn't load-bearing. Color carries meaning.
Typography signals function. Motion reveals data.

---

## 2. Typography

| Role | Family | Weights | Source | Used for |
|---|---|---|---|---|
| Display (sparing, editorial italic) | **Instrument Serif** | 400, 400i | Google Fonts | The word "flux" in the logo, the one-line plot caption ("*Signal* recovered through spiking dynamics"). Nothing else. |
| UI / body | **Inter Tight** | 300, 400, 500, 600, 700 | Google Fonts | All UI labels, section labels, captions. The default for everything sans-serif. |
| Mono / numerics | **JetBrains Mono** | 300, 400, 500, 600 | Google Fonts | **All numerals**, all data readouts, all keyboard-like elements (record IDs, mode names, timestamps), the meta-readout bar in the header, axis ticks. |

**Type rules:**
- Mono is for data. If a digit is shown, it lives in JetBrains Mono with
  `font-variant-numeric: tabular-nums` so columns of numbers align.
- Inter Tight is for prose / labels.
- Instrument Serif italic is the only "warm" typographic moment — use it for
  exactly two things: the `flux` in the logotype, and the editorial italic
  caption on the plot card. Anywhere else it dilutes the effect.

```css
:root {
  --f-display: 'Instrument Serif', serif;
  --f-ui:      'Inter Tight', system-ui, sans-serif;
  --f-mono:    'JetBrains Mono', monospace;
}
```

**Section labels** use uppercase mono with wide tracking (`letter-spacing:
0.14em–0.16em`) — looks like silkscreen on a control panel.

---

## 3. Color tokens

All colors are CSS variables on `:root`. Wire them into Tailwind via
`tailwind.config.ts` `theme.extend.colors` (see §8 below).

### Surfaces — prussian blueprint paper scale

| Token | Hex | Use |
|---|---|---|
| `--void` | `#0F1B2E` | Page background (visible only at viewport edges). |
| `--surface-0` | `#122036` | Header bar, controls panel, main canvas background. |
| `--surface-1` | `#15243C` | Cards, sliders track inactive, dropdown trigger. |
| `--surface-2` | `#1A2C46` | Mode-toggle active half, button default state. |
| `--surface-3` | `#213658` | Hover states, dropdown items hover. |

### Hairlines — drawn-in-chalk on blueprint

| Token | Hex | Use |
|---|---|---|
| `--rule` | `#2B3C5A` | Default 1px borders, grid lines in plots, dividers between metric cells. |
| `--rule-strong` | `#3D5378` | Borders around interactive elements (dropdown, button). |
| `--rule-hot` | `#557099` | Hover/active borders, brand-mark border. |

### Text — warm parchment ink

| Token | Hex | Use |
|---|---|---|
| `--text-1` | `#EDE4D3` | Primary — values, headlines, active states. The "ink." |
| `--text-2` | `#B5AB95` | Secondary — labels, captions. |
| `--text-3` | `#7E7660` | Tertiary — section labels (small caps), units, axis ticks. |
| `--text-4` | `#4A5670` | Quaternary — slider ticks, decorative dots. |

### Brand accents (semantic — each color has ONE job)

| Token | Hex | Semantic role |
|---|---|---|
| `--rust` | `#C04E2C` | **Reconstructed ECG trace.** Spike colour for **Layer 2 (hidden)**. **Positive ΔSNR.** The "NeuroFlux output" colour, the brand's primary accent. Reads as rust grease-pencil annotation on a schematic — the engineer's emphasis colour. |
| `--sage` | `#88A57B` | **Noisy ECG trace.** Spike colour for **Layer 1 (input)**. Reads as "muted/dampened/wrong" — like oxidized stain on the blueprint, deliberately not the brand colour. |
| `--mustard` | `#D4C158` | **Negative ΔSNR only.** Replaces the red/green cliché. Highest possible complementary contrast against prussian blue — pops as a warning without alarm. |
| `--parchment` | `#EDE4D3` | Spike colour for **Layer 3 (output)**. Same as `--text-1` — connotes "the decision layer is rendered in pure ink." |
| `--clean` | `#B5AB95` | The clean reference ECG trace. Deliberately understated — appears as a softly-drawn chalk line on the paper, ground truth not the hero. Same value as `--text-2`. |

**Why no red/green for ΔSNR sign:** clinical UIs over-use red/green and
evaluators in cardiology see them as alert signals (red = bad heart). Rust
positive + mustard negative reads as "annotation marks on a technical
drawing" — scientific rather than alarming. Rust also doubles as the brand
accent, reinforcing "positive Δ = NeuroFlux working."

**Layer color logic:** input layer is sage (the noisy unprocessed raw),
hidden is rust (processing/the brand emphasis), output is parchment
(the pure decision in ink). The spike raster stacks bottom→top (L1 sage to
L3 parchment), so the eye reads "muddied input rises through the rust-marked
processing layer into pure ink decisions."

**Why these colours work against prussian:**

| Pairing | Why |
|---|---|
| Rust orange on prussian blue | Classic complementary contrast (orange ↔ blue). Maximum perceived saturation. Familiar visual language: construction signs, antique blueprints, traffic signage. |
| Sage green on prussian blue | Soft contrast, reads as natural/quiet. Doesn't compete with the rust brand. |
| Mustard yellow on prussian blue | Yellow vs blue = highest possible complementary contrast. Forces attention to negative-Δ states. |
| Parchment cream on prussian | The blueprint signature. Maximal readability for text and data. |

---

## 4. Spacing

8px-based scale, with two intermediate values:

```css
--s1: 4px;  --s2: 8px;  --s3: 12px;
--s4: 16px; --s5: 24px; --s6: 32px;
--s7: 48px; --s8: 64px;
```

**Density:** This is an instrument. Padding is tighter than a typical
marketing site (controls panel uses `s5` outer padding, sections use `s5`
vertical). Don't airy-fy it.

---

## 5. Motion

Two custom easings — both feel "physical" rather than bouncy:

```css
--ease-out:    cubic-bezier(.22, 1, .36, 1);
--ease-in-out: cubic-bezier(.65, 0, .35, 1);
```

**Choreography of "Run simulation" (this is the demo's hero moment):**

| t (ms) | Element | Behavior |
|---|---|---|
| 0 | Run button | Pulse, then morph glyph into spinner |
| 600 | (sim returns from backend) | |
| 600 | Noisy trace | `drawOn` 900ms ease-out (left → right) |
| 700 | Clean trace | `drawOn` 900ms ease-out |
| 850 | Reconstructed trace | `drawOn` 900ms ease-out — appears last so it "lands" on top |
| 1000+ | R-peak markers (clean, then recon) | Fade-in scaled by index (20ms stagger) |
| 600 | Spike raster events | All fade in over 1000ms, staggered by their `t` position (later spikes appear later — the raster looks like it's being scanned out) |
| 600 | Metrics numbers | Animate count from previous value to new (600ms, ease-out cubic) |

The **trace-drawing** is the killer effect. It uses
`stroke-dasharray = pathLength; stroke-dashoffset: pathLength → 0`. The
reconstructed trace draws last and brightest, so the evaluator's eye lands on
it just as the metrics finish counting up.

**Mode toggle (Passive ↔ Trained):**
- Indicator slides 280ms `ease-in-out`.
- Metrics count from old to new values in 600ms.
- Trace re-renders smoothly (no flash — re-render to a new path then animate stroke).

**Hover playhead:**
- Vertical 0.8px dashed line follows mouse across plot.
- Readout in top-right (mono, glassmorphism backdrop-blur).
- 0ms latency — direct mouse follow, no transitions.

**Don't animate:**
- Hover states beyond color transitions (160ms ease-out).
- Layout. Everything is in fixed positions; nothing reflows on state change.
- Slider handles. The fill bar grows directly with input — no easing inside drag.

---

## 6. Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ Header bar (56px)                                                    │
│ ┌─────────┬─────────────────────────────────────┬─────────────────┐  │
│ │ brand   │ meta-readout (centered, mono)       │ status + tour   │  │
│ └─────────┴─────────────────────────────────────┴─────────────────┘  │
├─────────────┬────────────────────────────────────────────────────────┤
│             │                                                        │
│ Controls    │ ┌──────────────────────────────────────────────────┐   │
│ panel       │ │  HERO PLOT  (signal traces + R-peak markers)     │   │
│ (320px)     │ │                                                  │   │
│             │ │  [legend top-right]                              │   │
│ - source    │ │                                                  │   │
│ - noise     │ │  [playhead-readout bottom-right on hover]        │   │
│   freq/amp/ │ ├──────────────────────────────────────────────────┤   │
│   snr       │ │  SPIKE RASTER  (same time axis, 200px)           │   │
│ - learning  │ └──────────────────────────────────────────────────┘   │
│   epochs/   │ ┌────┬────┬────┬────┬────┐                             │
│   tau       │ │ ΔSNR│Rpeak│SNRo│Fire│ RT │  metrics row              │
│ - mode      │ └────┴────┴────┴────┴────┘                             │
│   toggle    │ ┌──────┬───┬───┬───┬───┬───┐                           │
│             │ │ count│ run │ run │ run │ ...   history strip         │
│ [RUN BTN]   │ └──────┴───┴───┴───┴───┴───┘                           │
└─────────────┴────────────────────────────────────────────────────────┘
```

### Layout rules
- **The stack** (plot + raster) shares one bordered card. They are NOT two
  cards. The shared border + matching x-axis dramatize the time alignment.
- The **legend** floats over the plot (top-right) with a backdrop blur. It is
  not in a separate panel — it overlays the data.
- Metrics and history are separate cards, each demarcated by hairlines.
- Min viewport: **1280px wide**. At 1024px the controls panel may need to
  collapse to 280px (drop the section index numbers); below that, don't
  support.

### Grid (Tailwind-friendly)
- Main app: `grid-cols-[320px_1fr] grid-rows-[56px_1fr]`.
- Stack: `grid-rows-[1fr_200px]` inside the bordered card.
- Metrics: `grid-cols-5 gap-px bg-[--rule]` (the 1px gap + dark background creates inter-cell hairlines).
- History: `grid-cols-[120px_repeat(5,1fr)] gap-px bg-[--rule]`.

---

## 7. Components

### 7.1 Header brand mark
- 24×24 bordered square with a hand-drawn sine wave SVG inside, stroked in rust.
- The wordmark uses a typographic split: **NEURO** in Inter Tight uppercase
  (semibold, tracking 0.18em, parchment) + **flux** in Instrument Serif italic
  lowercase, **rust**. This single typographic moment carries the brand voice.

### 7.2 Meta-readout (header center)
Single line of mono text showing the current config + a blinking rust
cursor. **Pattern:** `K v · K v · K v` where keys are tertiary text, values
are primary text, separated by `·` dot characters in `--text-4`. This is the
single most "instrument-like" element — it tells the evaluator exactly what
they're looking at at any moment.

### 7.3 Status pills (header right)
Small bordered pills with a colored LED dot. LED uses `box-shadow: 0 0 6px
<color>` for the glow effect. Two pills: NUMBA warm/cold, API port. Plus a
ghosted "TAKE THE TOUR" button.

### 7.4 Section header (controls)
- Small-caps mono label, left.
- Mono index number (`01`, `02`, ...) right in `--text-4`.
- The index numbers are a deliberate "instrument panel" affect — they make
  it feel like a numbered control board.

### 7.5 Slider
**This is the most distinctive control.** Do NOT use a generic library
slider — build it.
- Track: 2px tall, `--surface-3` background.
- Fill: 2px tall, `--rust` with `box-shadow: 0 0 8px rgba(192,78,44,0.4)` for glow (reads as "warm ink imprint").
- Handle: 12×12px diamond (rotated square), rust, no border.
- Tick marks below: 5 vertical 3px hairlines in `--text-4`.
- Header: name on left, mono value with unit on right.
- Optional hint after name in tiny mono `--text-4` (`τ` slider shows "trace decay").

When implementing in React, suggest `@radix-ui/react-slider` as the
behavior + accessibility primitive, then style it from scratch — Radix gives
keyboard/ARIA for free without imposing visuals.

### 7.6 Dropdown
- Trigger: mono record ID + small mono badge ("MIT-BIH · 30 min"), chevron right.
- List: dark `--surface-2` with stronger border, max-height with scroll.
- Items: hover background = `--surface-3`, hover text = rust.
- Use `@radix-ui/react-popover` or `react-aria-components` for accessibility,
  styled per the mockup.

### 7.7 Mode toggle (PASSIVE / TRAINED)
- Pill split into two equal halves.
- A sliding indicator (`--surface-3` background) moves between halves with a
  rust border on the leading edge.
- Each half: mono name (uppercase) + tiny meta caption ("0 ep · STDP off" /
  "10 ep · STDP on").
- **Below the toggle**: a one-line caption that changes with the mode and
  cites the Phase 3 finding (`rpeak = 1.000` passive). This is the
  scientific punchline — make sure it's always visible.

### 7.8 Run button
- 52px tall, full sidebar width.
- Default: `--surface-2`, border `--rule-hot`, mono uppercase label, play glyph.
- Hover: border rust, label rust, inner rust glow.
- Running: morph play-glyph into a spinning ring (CSS `border-top-color:
  transparent` + rotate animation), label becomes "Running…".

### 7.9 Hero plot
- Custom SVG, NOT Plotly. Plotly's default styling is the give-away.
- Stroke widths: noisy 1.1, clean 1.3, reconstructed 1.7. Reconstructed has
  a rust drop-shadow filter for the glow (`drop-shadow(0 0 5px rgba(192,78,44,0.35))`) —
  reads as "ink bleed on the paper."
- R-peak markers: small triangles (8×7 px) pointing up, 10px above the peak.
  Clean = outlined in `--clean`; reconstructed = filled `--rust`. They land
  AFTER the traces finish drawing.
- Grid: 4 horizontal + 10 vertical hairlines in `--rule`. Axis ticks in mono.
- **Subtle scanline overlay**: 1px transparent / 1px rgba-white(0.012)
  repeating gradient over the whole plot card. Almost invisible but adds
  "this is a screen" texture.
- **Plot caption**: top-left, two lines:
  - "HERO TRACE · TIME-DOMAIN RECONSTRUCTION" (mono, small caps)
  - "*Signal* recovered through spiking dynamics" (Instrument Serif italic, ~22px)

### 7.10 Spike raster
- 200px tall area below the plot, sharing the same `x` scale.
- 56 horizontal "rows" (one per neuron), with hairline rules separating layers.
- Each spike: 2×4px vertical rect, colored by layer (sage / rust / parchment).
- Layer labels on RIGHT edge in mono small caps: `L1 · INPUT · 32`, etc.
- Spikes fade in staggered by their `t` position — the raster looks like
  it's being scanned out, not just appearing. **This is the second most
  distinctive moment after the trace draw-on.**

### 7.11 Metrics row
- 5 equal cells, 1px gap (hairline separator effect via gap + dark background).
- Per cell:
  - Mono small-caps label
  - Huge mono number (28px) with smaller unit in `--text-3`
  - Sub-line or progress bar
- ΔSNR cell uses `--rust` if positive, `--mustard` if negative. **Never use
  red or green.**
- rpeak cell includes a thin bar (3px) filling proportional to value (1.000
  = full); bar uses rust if ≥ 0.85, mustard if below.
- Numbers should **count up** when results arrive (animate from previous
  value to new over 600ms with cubic ease-out).

### 7.12 History strip
- 1 header cell (120px) + 5 run cells, hairline separated.
- Header cell shows session run count in big mono.
- Run cells: timestamp · ΔSNR (with sign + color) · config summary in mono.
- Hover: surface-1 → surface-2 background.
- Click: in React, dispatch `restoreParams(entry)` — does NOT auto-run.

---

## 8. Tailwind configuration

Wire the design tokens into Tailwind so the friend can use semantic class
names instead of hex codes:

```ts
// packages/web/frontend/tailwind.config.ts
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
```

Set `font-ui` as the default body font in `src/styles/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body { @apply font-ui bg-void text-ink-1; }
}
```

Then components use semantic classes: `className="bg-surface-1 border border-rule-strong text-ink-2 font-mono"`.

---

## 9. Charting library — decision

**Decision: build the plot and spike raster in raw SVG (no chart library).**

| Considered | Verdict |
|---|---|
| Plotly.js | Rejected. Default styling is the AI-slop give-away. Restyling it to match this design takes more effort than building from scratch. |
| Recharts / Chart.js | Rejected. Same problem — opinionated defaults fight the design. |
| **uPlot** | Acceptable fallback if performance becomes an issue, but its theming surface is also limited. |
| **D3 selections** | Overkill for two static plots; the value-add is in scale + axis utilities, not rendering. |
| **Raw SVG + small helpers** (chosen) | Total control over every stroke. The mockup proves it works in ~200 lines. Encode reusable scale functions (`xScale`, `yScale`) as plain TS helpers. |

For React, encapsulate as `<SignalPlot data={...} />` and
`<SpikeRaster events={...} />` components that take typed props and render
SVG. Use `useResizeObserver` to compute a responsive `width`; height is
fixed (per the layout).

If you change your mind later: the API contract treats the plot as opaque,
so swapping is one-file replacements.

---

## 10. Component primitives — recommend Radix

Install `radix-ui` primitives for behavior/accessibility, style them per the
mockup:

```bash
npm i @radix-ui/react-slider \
       @radix-ui/react-select \
       @radix-ui/react-toggle-group \
       @radix-ui/react-tooltip \
       @radix-ui/react-dialog
```

**Do NOT** install shadcn/ui's default-themed components. Their defaults
contradict the instrument aesthetic (rounded corners, soft shadows, neutral
greys). Use Radix raw and style from scratch.

---

## 11. Motion library

**Use Framer Motion (`framer-motion`)** for the counting number animations
and any state-driven transitions. The trace draw-on is a CSS keyframe
animation against SVG `stroke-dashoffset` — no library needed.

```bash
npm i framer-motion
```

Animation budget: don't add motion that wasn't specified in §5. More
animation is not better. The Run-button choreography is the focal moment;
everything else should feel subtle.

---

## 12. The killer details (preserve these or it's just another dark dashboard)

If your friend cuts corners on these, the design loses its character:

1. **The trace draw-on.** Reconstructed last and brightest. Rust drop-shadow ("ink bleed on paper").
2. **The number-counting metrics.** Not a fade. An actual numeric tween.
3. **The blinking rust cursor** at the end of the meta-readout. 1.1s steps.
4. **The faint warm-cream scanline overlay** on the plot card. Almost invisible — reads as paper grain.
5. **The page-wide film grain** (SVG noise) at `opacity: 0.4 mix-blend overlay`.
6. **The mode-toggle caption** that cites the Phase 3 finding — make the
   `rpeak = 1.000` part rust.
7. **The hairline gap-as-border trick** for the metrics row (1px gap with
   `bg-rule` = inter-cell rules without box-shadow).
8. **JetBrains Mono with tabular-nums** for ALL numerics. Numbers don't dance
   when their values change.
9. **R-peak markers landing AFTER the traces draw.** Sequence matters.
10. **The brand split**: `NEURO` semibold uppercase (parchment) + `flux` Instrument
    Serif italic lowercase **rust**. The single most distinctive typographic moment.

---

## 13. What's intentionally NOT in this design

- **Dark/light mode toggle.** This is a scientific instrument — it has one
  mode. Don't add light mode.
- **Icons everywhere.** The design uses ~2 icons (brand mark, play glyph,
  chevron). Adding a Lucide icon to every button dilutes the typographic
  language.
- **Card shadows.** Everything is hairline-separated. No `shadow-lg`.
- **Rounded corners.** Inputs and cards have 0px or 2px (dropdown list) only.
  Sharp edges support the instrument feel.
- **Gradients.** The only gradient is the radial atmosphere on `body`
  (extremely subtle warm rust + sage washes). No gradient buttons, no gradient text.
- **Purple.** The classic AI-slop accent. Forbidden.
- **Cyan + dark.** The default 2024–2025 AI-generated dashboard combo.
  Forbidden — that's the entire reason we picked this palette over the
  original direction.
- **Tooltips on every control.** The mockup uses inline `.hint` text where
  context is needed; tooltips are reserved for sliders' parameter
  explanations from `CLAUDE.md` / `docs/phase3_findings.md`.

---

## 14. Implementation checklist for the friend

When you implement a component, verify against this list before merging:

- [ ] Fonts: Instrument Serif (display, italic only), Inter Tight (UI),
      JetBrains Mono (numerics).
- [ ] Numbers use `font-variant-numeric: tabular-nums`.
- [ ] Hairlines are 1px solid `--rule` (or `--rule-strong` for interactive
      borders). No box-shadows.
- [ ] Colors used semantically: rust = reconstructed/positive/L2; sage =
      noisy/L1; mustard = negative ΔSNR; parchment = L3; `--clean` = clean trace.
- [ ] Mono small-caps labels with `letter-spacing: 0.14em–0.16em` and
      `font-size: 10–11px`.
- [ ] Hover transitions are 160ms `ease-out` (or use `--instrument-out`).
- [ ] No purple anywhere.
- [ ] No rounded corners on buttons/inputs beyond 0–2px.
- [ ] No shadows on cards.

---

## 15. References

- `mockup.html` — interactive proof, source of truth for visuals.
- `PHASE5_IMPLEMENTATION.md` §3.2 — frozen tech stack.
- `docs/phase3_findings.md` §5.6.2 — the passive-rpeak-1.000 finding the
  mode toggle dramatizes.
- Instrument Serif: <https://fonts.google.com/specimen/Instrument+Serif>
- Inter Tight: <https://fonts.google.com/specimen/Inter+Tight>
- JetBrains Mono: <https://fonts.google.com/specimen/JetBrains+Mono>
- Radix primitives: <https://www.radix-ui.com/primitives>
