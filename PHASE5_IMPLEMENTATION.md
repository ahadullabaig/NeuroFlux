# PHASE5_IMPLEMENTATION.md

A self-contained execution guide for **Phase 5 — Enhanced interactive dashboard**.
Read this end-to-end before writing any code. Every architectural decision and
task ordering has been justified here so the executor doesn't re-litigate.

**If this is the first time you've worked in this repo on this machine, do
§0 (First-time setup) before anything else.** The plan was written by someone
whose environment was already set up; §0 reconstructs that environment from a
fresh clone.

> Companion docs:
> - `packages/web/design/DESIGN.md` — **visual design system + token reference. Read this before any UI code.**
> - `packages/web/design/mockup.html` — interactive proof-of-design; open in a browser to see what you're building.
> - `PLAN.md` §Phase 5 (lines 384-419) for the original spec.
> - `docs/phase3_findings.md` for the science the dashboard is showcasing.
> - `CLAUDE.md` for repo conventions.

---

## 0. First-time setup (do this before §1)

This plan assumes a fresh clone on a machine that has **never run NeuroFlux
before**. Several artefacts (MIT-BIH cache, `.venv`, generated checksums) live
outside git and must be re-created locally. Run the steps below once, in order.

### 0.1 System prerequisites

Install at the OS level (these are not in the repo):

| Tool | Version | How to check | Where to get |
|---|---|---|---|
| Git | any recent | `git --version` | your OS package manager |
| Python | **3.11.x** (the repo pins this) | `python3 --version` | https://www.python.org/downloads/ or pyenv |
| `uv` | ≥ 0.4 | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | **≥ 20 LTS** | `node --version` | https://nodejs.org/ or nvm |
| npm | ≥ 10 (ships with Node 20) | `npm --version` | bundled with Node |

Optional but useful: `jq` (for the curl examples in this doc), `sqlite3` (for
poking at `results/*.db`).

### 0.2 Clone and Python install

```bash
git clone <repo-url> unisys_project
cd unisys_project
git checkout phase-5                  # or whichever branch holds this plan
uv sync --all-packages                # creates .venv/ and installs editable
                                      # packages from the workspace
```

If `uv sync` fails on `wfdb`, ensure Python is 3.11.x (not 3.12+) — the wfdb
wheel matrix lags newer Python releases.

### 0.3 Download MIT-BIH records (REQUIRED for the demo)

The dashboard's record dropdown is populated from records found in
`data/mit_bih/`. **The `data/` directory is gitignored** — none of the
`.dat`/`.hea`/annotation files are in the repo. They must be downloaded
locally. The committed manifest already has SHA-256 checksums pinned for the
16 records Phase 3 used, so the downloads are integrity-verified
automatically.

Run this from the repo root after `uv sync`:

```bash
uv run python - <<'PY'
from neuroflux_datasets.mit_bih.downloader import download_record

# The 16 records Phase 3 used. Each is ~1–5 MB; total ~50 MB.
records = [
    "100", "101", "103", "105", "108", "112", "113", "119",
    "200", "207", "212", "214", "228", "231", "232", "234",
]
for rid in records:
    print(f"Downloading {rid}...")
    download_record(rid, "data/mit_bih")
print("Done.")
PY
```

Expected output: 16 "Downloading <id>..." lines, finishing with "Done." and no
exceptions. If a `ChecksumError` raises, the manifest expects a specific
digest and the download server returned different bytes — investigate before
retrying.

Verify the cache:

```bash
ls data/mit_bih/*.dat | wc -l            # should print 16
```

If you need a smaller demo (fewer records), at minimum download **214** — the
Phase 3 canonical positive config uses it, and Milestone C's manual
acceptance test selects it specifically:

```bash
uv run python -c "from neuroflux_datasets.mit_bih.downloader import download_record; download_record('214', 'data/mit_bih')"
```

### 0.4 Verify the engine works before touching the dashboard

Before any Phase 5 code, confirm the existing engine and Phase 3 regression
pass on this machine. This catches platform-specific issues early (numerics
diverge on Apple Silicon vs. x86, etc.).

```bash
make test-fast                        # ~30 s; runs the CI test suite
NUMBA_DISABLE_JIT=1 uv run pytest tests/datasets/test_phase3_regression.py -v
# Expect either: PASSED (if record 214 was downloaded above)
# Or:           SKIPPED (if you skipped 0.3 — but you'll need it later)
```

If `make test-fast` fails on a clean clone, **stop and fix the environment
before starting Phase 5**. The dashboard reuses these code paths; failures
here will resurface as silent demo bugs.

### 0.5 Files / directories the executor will create that are NOT in git

For reference — none of these need to be created manually before starting
Phase 5; the per-task steps below cover their creation. Listed here so the
executor knows what's expected to appear on a fresh clone.

| Path | Created by | Gitignored? |
|---|---|---|
| `.venv/` | `uv sync` (step 0.2) | yes |
| `data/mit_bih/*.dat`, `*.hea`, etc. | downloader (step 0.3) | yes |
| `data/mit_bih/*.provenance.json` | downloader (step 0.3) | yes |
| `packages/web/backend/` and everything inside | Milestone A | no |
| `packages/web/frontend/node_modules/` | `npm install` (B1) | yes (add in B1) |
| `packages/web/frontend/dist/` | `npm run build` | yes (add in B1) |
| `packages/web/frontend/playwright-report/`, `test-results/` | `npm run test:e2e` | yes (add in B1) |
| `tests/web/` directory | Milestone A test files | no |
| `results/*.db`, `*.log`, etc. from any ad-hoc runs | not used by Phase 5 | yes (already) |

### 0.6 Two-shell layout during development

You will need two terminals open most of the time:

| Shell | Command (run from repo root) | Purpose |
|---|---|---|
| 1 | `uv run uvicorn neuroflux_web.app:create_app --factory --reload --port 8000` | Backend API on `:8000` |
| 2 | `cd packages/web/frontend && npm run dev` | Frontend on `:5173` |

Optionally a third shell for tests / commits.

---

## 1. Goal

Ship a polished, evaluator-facing web dashboard that:

1. Lets a reviewer load a real ECG signal (cached MIT-BIH record OR uploaded
   CSV), inject configurable flicker noise, run NeuroFlux, and see the
   reconstructed signal with R-peaks overlaid in under 5 seconds end-to-end.
2. Tells the **Phase 3 story** visually: passive vs. trained NeuroFlux toggle
   exposes the most defensible finding (passive `rpeak=1.000` baseline).
3. Looks professional. This is a presentation surface, not a lab notebook.

**Non-goal:** ML/optimisation of the SNN itself. Phase 5 is a UI layer; the
engine is frozen.

---

## 2. Scope (in vs. out for v1)

### In scope for v1
- Backend FastAPI service wrapping `neuroflux_core` + `neuroflux_datasets`,
  **local-dev only** (no hosting, no auth, no telemetry — see §9).
- Frontend React + Vite + TypeScript SPA.
- Synchronous `/simulate` endpoint (no streaming; 10 s signals run in < 2 s).
- MIT-BIH record dropdown (16 cached records).
- CSV upload (single-channel ECG, configurable fs).
- Flicker controls (frequency, amplitude, input SNR).
- Passive vs. trained NeuroFlux toggle (epochs 0 vs. user-selected).
- Live signal plot (clean + noisy + reconstructed overlay, R-peak markers).
- Spike raster visualisation.
- Metrics panel (input SNR, output SNR, ΔSNR, rpeak\_integrity, mean firing
  rate).
- CSV export of reconstructed signal.
- LocalStorage-based experiment history (last 20 runs).
- Narrative-mode guided tour.
- Playwright smoke test of the golden-path flow.
- Developer-mode README (how to run locally).

### Out of scope for v1 (justify if changing)
- **Baseline comparison view** — requires Phase 4 (`packages/baselines/`). Stub
  panel with "Phase 4 — coming soon".
- **EDF / WFDB upload** — CSV first. EDF/WFDB add ~1 day of file-parsing edge
  cases and aren't blocking the demo. Defer to v1.1.
- **WebSocket streaming** — only needed for > 30 s signals. v1 demos are 10 s.
  Spec'd in §Milestone F as optional.
- **Hosted deployment (Vercel/Netlify)** — backend is a Python service; hosting
  it requires infra decisions outside this phase. v1 is local-dev only.
  Deployment is a follow-up task.
- **Mobile breakpoint < 768 px** — tablet (≥ 768 px) is enough for demos.
- **WebGPU acceleration** — listed as PLAN.md stretch goal; not v1.
- **In-browser SNN port (JS/WASM)** — see §3.1 below.

### Deviation from PLAN.md acceptance criteria

PLAN.md:416 states: *"Complete workflow: user uploads an ECG file, runs
NeuroFlux, downloads filtered output — entirely in browser, no server."*

**v1 deviates: we ship a Python backend.** Rationale in §3.1. The "no-server"
property remains a long-term goal but is dropped from v1 to preserve numerical
fidelity to the Phase 3 results. Document this in `README.md` when v1 lands.

---

## 3. Architectural decisions (already made — do not re-litigate)

### 3.1 Backend bridge vs. in-browser SNN

**Decision: FastAPI backend bridge to the existing Python engine.**

| Aspect | In-browser (JS/WASM) | Backend bridge (chosen) |
|---|---|---|
| Engine source of truth | Re-implement in JS | Re-use `neuroflux_core` as-is |
| Numerical fidelity vs. Phase 3 | Risks divergence (separate Numba/JS RNG, separate STDP code path) | Identical (same code) |
| Effort | 2–3 weeks JS rewrite + validation | 2–3 days FastAPI wrapper |
| Risk if evaluators cross-check claims | High (subtle off-by-one shifts ΔSNR) | None |
| Hosting | None | Needs a Python host |
| Phase 3 regression tests | Need a separate JS regression suite | Existing pytest suite covers it |

Evaluators are likely to spot-check numerics against `docs/phase3_findings.md`.
Fidelity wins.

### 3.2 Tech stack (frozen)

| Layer | Choice | Why |
|---|---|---|
| Backend framework | FastAPI | Async, Pydantic v2 native (matches existing configs), OpenAPI auto-generation |
| ASGI server | uvicorn | Standard pairing |
| Python deps | uv workspace | Existing tooling |
| Frontend bundler | Vite | Fast dev server, well-trodden React+TS template |
| UI library | React 18 + TypeScript | Conventional, hiring market, large ecosystem |
| Styling | Tailwind CSS **v3.4** (pin — v4 is a different config model) | Tokens wired in via `tailwind.config.ts` (see `packages/web/design/DESIGN.md` §8) |
| State | Zustand | Hooks-based, no boilerplate, scales to v1.x without rewrite |
| Charting | **Raw SVG + small TS helpers — NO chart library** | See `DESIGN.md` §9. Plotly's defaults contradict the "instrument" aesthetic; restyling it costs more than building plot + raster from scratch (~200 lines each, proven in `mockup.html`). |
| Component primitives | **Radix UI** (`@radix-ui/react-slider`, `-select`, `-toggle-group`, `-tooltip`, `-dialog`) | Headless behavior + accessibility, styled from scratch per the design. **Do NOT install shadcn/ui** — its defaults fight the design language. |
| Motion | **Framer Motion** (`framer-motion`) | For counting-number tweens and state transitions. The trace draw-on is a pure CSS keyframe (no library needed). |
| Fonts | **Instrument Serif** (display, italic only), **Inter Tight** (UI), **JetBrains Mono** (numerics) — all Google Fonts | Committed in `DESIGN.md` §2 |
| HTTP client | Native `fetch` | One dependency less |
| Package manager (frontend) | **npm** (pin — keep tooling assumptions simple) | |
| E2E test | Playwright | Modern, easy local + CI usage |

### 3.3 Repository layout (target)

```
packages/web/
├── backend/                                # uv workspace member
│   ├── pyproject.toml
│   └── src/neuroflux_web/
│       ├── __init__.py
│       ├── app.py                          # FastAPI app factory + CORS
│       ├── models.py                       # Pydantic request/response models
│       ├── services/
│       │   └── simulator.py                # Wraps neuroflux_core Simulator
│       └── routes/
│           ├── __init__.py
│           ├── health.py
│           ├── records.py                  # GET /api/records, /api/records/{id}
│           ├── simulate.py                 # POST /api/simulate
│           ├── upload.py                   # POST /api/upload (CSV)
│           └── export.py                   # GET /api/export/{sim_id}.csv
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── postcss.config.js
    ├── index.html
    ├── playwright.config.ts
    ├── tests/
    │   └── e2e/golden_path.spec.ts
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/
        │   ├── client.ts                   # fetch wrapper
        │   └── types.ts                    # mirrors backend models
        ├── components/
        │   ├── ControlsPanel.tsx
        │   ├── SignalPlot.tsx
        │   ├── SpikeRaster.tsx
        │   ├── MetricsPanel.tsx
        │   ├── HistoryPanel.tsx
        │   ├── UploadDropzone.tsx
        │   ├── NarrativeTour.tsx
        │   └── Layout.tsx
        ├── store/
        │   └── simulationStore.ts          # Zustand store
        └── styles/
            └── index.css                   # Tailwind directives
tests/web/                                  # Python backend tests
├── conftest.py
├── test_health.py
├── test_records.py
├── test_simulate.py
├── test_upload.py
└── test_export.py
```

### 3.4 API contract (frozen — backend implements, frontend consumes)

All paths prefixed `/api/`. JSON bodies. Versioned implicitly via `/api/v1/` —
**do not add `v1` yet**; introduce when a breaking change lands.

| Method | Path | Body / params | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{ status: "ok", numba_warm: bool }` |
| GET | `/api/records` | — | `[{ record_id, channels, fs, duration_s }]` |
| GET | `/api/records/{id}/segment?duration_s=10.0` | — | `{ record_id, fs, signal: number[] }` |
| POST | `/api/upload` | multipart CSV + `fs` form field | `{ signal_id, fs, n_samples }` (temp store) |
| POST | `/api/simulate` | `SimulateRequest` (see §3.5) | `SimulateResponse` (see §3.5) |
| GET | `/api/export/{simulation_id}.csv` | — | CSV file (text/csv) |

### 3.5 Pydantic models (frozen)

```python
# packages/web/backend/src/neuroflux_web/models.py
class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["record", "upload"]
    record_id: str | None = None                # required if source=="record"
    signal_id: str | None = None                # required if source=="upload"
    duration_s: float = Field(default=10.0, gt=0, le=30.0)
    flicker_freq_hz: float = Field(default=50.0, ge=0.0, le=200.0)
    flicker_amplitude: float = Field(default=0.5, ge=0.0, le=2.0)
    input_snr_db: float = Field(default=0.0, ge=-10.0, le=30.0)
    training_epochs: int = Field(default=10, ge=0, le=100)
    stdp_tau_ms: float = Field(default=10.0, gt=0, le=100.0)
    network_layers: list[int] = Field(default_factory=lambda: [32, 16, 8])
    seed: int = 0

class SimulateResponse(BaseModel):
    simulation_id: str                          # uuid4 hex
    fs: float
    t_ms: list[float]                           # length N (subsample if > 5000)
    clean: list[float]
    noisy: list[float]
    reconstructed: list[float]
    rpeaks_clean_ms: list[float]
    rpeaks_reconstructed_ms: list[float]
    spike_events: list[SpikeEvent]              # sparse: (t_ms, layer, neuron)
    metrics: SimulateMetrics
    runtime_ms: float

class SpikeEvent(BaseModel):
    t_ms: float
    layer: int
    neuron: int

class SimulateMetrics(BaseModel):
    input_snr_db: float
    output_snr_db: float
    delta_snr_db: float
    rpeak_integrity: float
    mean_firing_rate_hz: float
```

Frontend mirrors these in `frontend/src/api/types.ts`. If you change a model,
change both sides in the same commit.

### 3.6 Operational gotchas to plan for

| Gotcha | Mitigation |
|---|---|
| Numba JIT cold start (~5 s on first `/simulate`) | App-startup hook runs a tiny warmup simulation. Health endpoint reports `numba_warm: true` once done. |
| CORS (frontend `:5173` → backend `:8000`) | `fastapi.middleware.cors.CORSMiddleware` allowing `localhost:5173` |
| Spike raster payload size (10 s × 8 neurons × ~20 Hz ≈ 1.6k events; manageable) | Already sparse. Cap at 5000 events; raise warning if exceeded. |
| Plotly performance | < 2500 points per trace is fine. If signal subsampled to fit, document the subsample factor in the response. |
| Tailwind v4 breakage | Pin v3.4 explicitly in `package.json`. v4 changed config from JS to CSS-only. |
| Type drift between Pydantic and TS | Manual mirror (small surface, ~5 models). If surface grows, switch to `openapi-typescript` codegen — out of scope for v1. |
| Numba env var in tests | Backend tests must run with `NUMBA_DISABLE_JIT=1`. Add to `tests/web/conftest.py` via `monkeypatch.setenv` *before* `neuroflux_core` import. Easiest: import inside test bodies, or set env var in `pyproject.toml` [tool.pytest.ini_options]. |
| Concurrent /simulate requests | v1: serialise with a global asyncio lock. Production-grade pooling is post-v1. |

---

## 4. Difficulty rubric and batching rules

Each step is flagged **Easy / Medium / Hard**.

| Tier | Sub-step count | Execution rule |
|---|---|---|
| Easy (E) | 1 sub-step | May be batched with other Easy tasks if no dependency conflict. |
| Medium (M) | 1–2 sub-steps | May be batched with up to one other Medium **or** several Easy tasks if the executor is confident in correctness. |
| Hard (H) | 3–4 sub-steps | **Executed individually.** No batching with other Hard tasks. Verify each sub-step before the next. |

After every batch:
1. Run the **per-step acceptance check** listed on each task.
2. Run the **batch verification** at the milestone end.
3. Stop and report before starting the next batch.

The executor uses judgement on Easy/Medium batching — confidence in absence of
bugs is the criterion, not throughput.

---

## 5. Task catalogue

### Milestone A — Backend foundation

#### A1 — Scaffold the `neuroflux_web` package &nbsp; [E]
**Sub-steps:** 1
- Create `packages/web/backend/pyproject.toml` (name `neuroflux-web`,
  dependencies: `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `python-multipart`,
  `neuroflux-core`, `neuroflux-datasets`, `pydantic>=2.0`, `numpy`).
- Add the package to `[tool.uv.workspace] members` in the root `pyproject.toml`.
- **Lift `packages/web` from the mypy `exclude` list** in root `pyproject.toml`
  so backend code is type-checked by CI. (Frontend is still excluded — mypy
  doesn't see `.ts` files.)
- Create empty `src/neuroflux_web/__init__.py`, `app.py`, `models.py`,
  `routes/__init__.py`, `services/__init__.py`.

**Files:** `pyproject.toml` (root), `packages/web/backend/pyproject.toml`,
`packages/web/backend/src/neuroflux_web/{__init__,app,models}.py`,
`packages/web/backend/src/neuroflux_web/routes/__init__.py`,
`packages/web/backend/src/neuroflux_web/services/__init__.py`.

**Acceptance:**
```bash
uv sync --all-packages           # resolves without error
uv run mypy packages/web/        # passes (empty package)
uv run python -c "import neuroflux_web"
```

---

#### A2 — Health endpoint with Numba warmup &nbsp; [E]
**Sub-steps:** 1
- In `app.py`, build a `create_app()` factory that:
  - Registers CORS middleware allowing `http://localhost:5173`.
  - Includes the `health` router.
  - On `@app.on_event("startup")`, runs a 1 ms simulation to JIT-compile LIF/STDP
    and sets `app.state.numba_warm = True`.
- Add `routes/health.py` exposing `GET /api/health` returning `{ "status": "ok",
  "numba_warm": app.state.numba_warm }`.
- Add `tests/web/test_health.py` using `httpx.AsyncClient` + FastAPI lifespan.

**Acceptance:**
```bash
uv run pytest tests/web/test_health.py -v
# In a separate shell:
uv run uvicorn neuroflux_web.app:create_app --factory --reload
curl http://localhost:8000/api/health  # returns {"status":"ok","numba_warm":true}
```

---

#### A3 — MIT-BIH records listing endpoint &nbsp; [E]
**Sub-steps:** 1
- `routes/records.py`:
  - `GET /api/records` → reads `packages/datasets/.../mit_bih/manifest.json`
    and returns metadata for records present in `data/mit_bih/` (filter to
    actually-cached records — check `.dat` file existence). For each record,
    include `{record_id, fs, duration_s}` (durations from manifest or `wfdb`
    lazily).
  - `GET /api/records/{id}/segment?duration_s=10.0` → uses
    `neuroflux_datasets.mit_bih.annotator.extract_clean_segments` to pull the
    first clean segment; returns `{record_id, fs, signal: list[float]}`.
- Test against the 16 cached records (skip individual `segment` tests if data
  missing — pattern used in `tests/datasets/test_phase3_regression.py`).

**Acceptance:**
```bash
uv run pytest tests/web/test_records.py -v
curl http://localhost:8000/api/records | jq '. | length'        # 16
curl 'http://localhost:8000/api/records/214/segment?duration_s=10' \
  | jq '.signal | length'                                       # 2500
```

> If `/api/records` returns `0` or `214/segment` 404s, `data/mit_bih/` is
> empty — go back and run §0.3.

---

#### A4 — Synchronous `/simulate` endpoint &nbsp; [M]
**Sub-steps:** 2

**A4.1 — Service layer**
- `services/simulator.py`:
  - `run_simulation(req: SimulateRequest, source_signal: np.ndarray, fs: float) -> SimulateResponse`
  - Mirrors what `neuroflux_datasets.cli.run_single_experiment` does: builds
    `Network.feedforward(layers=req.network_layers, seed=req.seed)`, wires a
    `Simulator`, runs flicker injection + Poisson encode + simulate +
    optionally STDP-train (skip training if `req.training_epochs == 0`) +
    evaluation pass.
  - Collects spikes per layer into sparse `SpikeEvent` list (cap at 5000;
    decimate uniformly if over).
  - Computes metrics via `calculate_snr` + `rpeak_integrity` + `mean_firing_rate`.
  - Returns the `SimulateResponse`.
- Unit-test with a synthetic signal: assert `len(t_ms) == len(clean) ==
  len(noisy) == len(reconstructed)`, `metrics.delta_snr_db` finite, no spike
  events past `duration_s * 1000`.

**A4.2 — Route + serialisation**
- `routes/simulate.py`:
  - `POST /api/simulate` accepts `SimulateRequest`.
  - If `source == "record"`: load via `records.py` helper.
  - If `source == "upload"`: fetch from in-memory upload store (see A5).
  - Acquires a global `asyncio.Lock` to serialise.
  - Calls `simulator.run_simulation`.
  - Stores `SimulateResponse` in an LRU upload-cache keyed by
    `simulation_id` (for `/api/export` to read later — TTL 1 hour, max 20).
- Tests cover: record path, upload path (mocked), invalid record id (404),
  duration > 30 s (422 from Pydantic).

**Acceptance:**
```bash
uv run pytest tests/web/test_simulate.py -v
curl -X POST http://localhost:8000/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"source":"record","record_id":"214","duration_s":10,
       "flicker_freq_hz":50,"input_snr_db":0,"training_epochs":10}' \
  | jq '.metrics'
# Expect delta_snr_db > 0 and rpeak_integrity > 0.7
```

---

#### A5 — CSV upload endpoint &nbsp; [M]
**Sub-steps:** 2

**A5.1 — Parser + storage**
- `routes/upload.py`: `POST /api/upload` accepts multipart
  `file: UploadFile` + `fs: float` form field.
- Parser: single-column CSV, optional header; reject if > 2 columns, > 100k
  rows, or non-numeric values. Returns `{signal_id, fs, n_samples}`.
- Storage: in-process `dict[str, tuple[float, np.ndarray]]` keyed by uuid4. TTL
  via simple timestamp + cleanup on access. Max 50 entries (LRU).

**A5.2 — Tests**
- Happy path: 1-column CSV with header.
- Reject: header-only, > 2 columns, non-numeric cell.
- TTL/LRU: insert 51 signals, oldest is evicted.

**Acceptance:**
```bash
uv run pytest tests/web/test_upload.py -v
echo "value\n1\n2\n3\n4\n5" > /tmp/test.csv
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/tmp/test.csv" -F "fs=250" | jq .
# Expect {signal_id: "...", fs: 250.0, n_samples: 5}
```

---

#### A6 — Export endpoint &nbsp; [E]
**Sub-steps:** 1
- `routes/export.py`: `GET /api/export/{simulation_id}.csv` reads from the LRU
  cache populated by A4.2, returns a `text/csv` response with columns
  `t_ms,clean,noisy,reconstructed`. 404 if expired/unknown.

**Acceptance:**
```bash
uv run pytest tests/web/test_export.py -v
SIM_ID=$(curl -sX POST .../simulate -d '...' | jq -r .simulation_id)
curl -s "http://localhost:8000/api/export/${SIM_ID}.csv" | head -3
```

**Milestone A verification (run before moving on):**
```bash
uv run ruff format --check packages/web/backend/
uv run ruff check packages/web/backend/
uv run mypy packages/web/
uv run pytest tests/web/ -v
make test-fast                 # ensure no regressions to existing tests
```

---

### Milestone B — Frontend foundation

#### B1 — Vite + React + TS scaffold &nbsp; [E]
**Sub-steps:** 1
- `npm create vite@latest packages/web/frontend -- --template react-ts`
- Pin `node` ≥ 20 in a comment in `package.json`.
- Add `.gitignore` for `node_modules/`, `dist/`, `playwright-report/`,
  `test-results/`. (Add these to root `.gitignore` if not present.)
- Verify dev server boots: `npm run dev` in `packages/web/frontend/`.

**Acceptance:** `npm run build` succeeds; default Vite splash renders at
`http://localhost:5173`.

---

#### B2 — Tailwind v3.4 + design tokens from DESIGN.md &nbsp; [E]
**Sub-steps:** 1
- `npm i -D tailwindcss@3.4 postcss autoprefixer`
- `npx tailwindcss init -p`
- **Copy `tailwind.config.ts` directly from `packages/web/design/DESIGN.md`
  §8.** All color tokens (`void`, `surface.0-3`, `rule[.strong|.hot]`,
  `ink.1-4`, `signal.*`, `accent.*`), font families, easings, and shadows
  are spec'd there. Do NOT invent your own palette.
- Add Google Fonts links to `index.html` for Instrument Serif (italic),
  Inter Tight (300–700), JetBrains Mono (300–600). Copy the `<link>` tag
  pattern from the `<head>` of `packages/web/design/mockup.html`.
- `src/styles/index.css`:
  ```css
  @tailwind base; @tailwind components; @tailwind utilities;
  @layer base { body { @apply font-ui bg-void text-ink-1; } }
  ```
  Import in `main.tsx`.
- Verify with the brand-mark in the header: `<span className="font-ui font-semibold tracking-[0.18em] uppercase text-[13px]">Neuro<span className="font-display italic text-[17px] text-accent-cyan normal-case tracking-normal">flux</span></span>` (see DESIGN.md §7.1).

---

#### B3 — Layout shell &nbsp; [E]
**Sub-steps:** 1
- `components/Layout.tsx`: implement the grid from DESIGN.md §6.
  - `grid-cols-[320px_1fr] grid-rows-[56px_1fr] h-screen`.
  - Header bar (56px) with brand left, meta-readout center, status pills + tour button right.
  - Left sidebar = `<aside>` for controls panel (320px).
  - Main canvas = `<main>` with `grid-rows-[1fr_auto_auto] gap-4 p-6`.
  - Add the SVG-noise grain overlay (`body::before` pattern in mockup.html).
- Replace `App.tsx` body with `<Layout>...</Layout>`.

**Acceptance:** Compare side-by-side with `mockup.html` — header sizes,
sidebar width, gap between cards should match within 4px.

---

#### B4 — Zustand store &nbsp; [M]
**Sub-steps:** 1
- `npm i zustand`
- `store/simulationStore.ts`:
  ```ts
  interface SimulationState {
    params: SimulateRequest;
    setParams: (patch: Partial<SimulateRequest>) => void;
    status: 'idle' | 'running' | 'error';
    response: SimulateResponse | null;
    error: string | null;
    runSimulation: () => Promise<void>;          // calls api/client
    history: HistoryEntry[];                      // populated in D4
    addToHistory: (entry: HistoryEntry) => void;
  }
  ```
- Persist `history` to LocalStorage via Zustand's `persist` middleware (cap at
  20 entries).

**Acceptance:** Store imports cleanly; minimal smoke component sets/reads
`params.input_snr_db`.

---

#### B5 — API client + types &nbsp; [M]
**Sub-steps:** 2

**B5.1 — Type definitions**
- `src/api/types.ts` mirrors backend Pydantic models exactly (SimulateRequest,
  SimulateResponse, SpikeEvent, SimulateMetrics, RecordSummary).

**B5.2 — Client functions**
- `src/api/client.ts` exports `listRecords`, `getRecordSegment`,
  `uploadCsv`, `simulate`, `exportCsvUrl(simulationId)`.
- All return strongly-typed responses; throw `ApiError` with status + message
  on non-2xx.
- Base URL from `import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'`.

**Acceptance:** From a temp component, calling `await listRecords()` populates
a dropdown — manual smoke test.

**Milestone B verification:**
```bash
cd packages/web/frontend && npm run build && npm run dev
# Open localhost:5173, see Tailwind-styled layout shell, no console errors
```

---

### Milestone C — Core demo flow (critical path; the v1 minimum)

#### C1 — Controls panel &nbsp; [M]
**Sub-steps:** 2

**C1.1 — Custom slider + section primitives**
- Build a `<Slider>` component wrapping `@radix-ui/react-slider`, styled
  per DESIGN.md §7.5: 2px track, glowing cyan fill, 12px diamond handle
  (rotated 45deg), 5 tick marks below, slider-head row with mono value
  + unit. Do NOT use the Radix default styles — strip them entirely
  (`[&_[data-orientation]]:bg-transparent` etc.) and rebuild.
- Build `<Section>` wrapper: section-label small caps + mono index number
  on the right; bordered with `border-b border-rule` between sections.
- Build `<Dropdown>` (Radix Select styled per §7.6) with mono record IDs and
  badge text.

**C1.2 — Compose ControlsPanel + mode toggle + run button**
- `ControlsPanel.tsx`:
  - Section 01 SOURCE: dropdown + (later in D3) upload tab.
  - Section 02 NOISE INJECTION: three sliders (flicker freq, amplitude, input SNR).
  - Section 03 STDP LEARNING: two sliders (epochs, τ).
  - Section 04 MODE: passive/trained toggle (§7.7) with the dynamic caption
    citing `rpeak = 1.000` for passive. Toggling preserves the user's epoch
    setting (store sets `epochs = 0` while in passive mode, restores on
    un-toggle).
  - Run button (§7.8) at the bottom with `marginTop: auto`.
- Two-way bind to Zustand store. Tooltip text on slider names: lift from
  `CLAUDE.md` (e.g. τ = "trace decay constant, controls STDP plasticity
  window").
- Disable Run if no source is set; keyboard-accessible (Radix handles ARIA).

**Acceptance:** Adjust sliders, click Run — store dispatches
`runSimulation()` (stub it for now). Toggle passive ↔ trained — caption
text updates, indicator slides 280ms ease-in-out, meta-readout in header
updates to show the new mode.

---

#### C2 — Signal plot with R-peak overlay &nbsp; [H]
**Sub-steps:** 4. Build from raw SVG per DESIGN.md §7.9 — no Plotly. The
mockup's `<svg id="plot-svg">` rendering in `mockup.html` is the reference
implementation; port it to React/TS.

**C2.1 — Scale helpers + grid**
- `src/components/plot/scales.ts`: `xScale(t, duration, width, margin)`,
  `yScale(v, ymin, ymax, height, margin)`, `buildPath(samples, fs, step)`.
- `<SignalPlot>` component skeleton: takes typed props
  `{ tMs: number[], clean: number[], noisy: number[], reconstructed: number[],
  rpeaksCleanMs: number[], rpeaksReconstructedMs: number[] }`.
- Renders the bordered card (`border border-rule bg-surface-1`), the plot
  caption (mono small-caps line + Instrument Serif italic line), and the
  hairline grid (4 horizontal + 10 vertical lines in `stroke-rule`).
- Use `useResizeObserver` for responsive width; height fixed.

**C2.2 — Traces with draw-on animation**
- Render three `<path>` elements with classes mapped to design tokens:
  `stroke-signal-noisy stroke-[1.1]`, `stroke-signal-clean stroke-[1.3]`,
  `stroke-signal-reconstructed stroke-[1.7] [filter:drop-shadow(0_0_5px_rgba(0,217,192,0.35))]`.
- Animate stroke-dashoffset from pathLength → 0 (900ms cubic-bezier
  `.22,1,.36,1`) staggered by 0/100/250ms (noisy / clean / reconstructed).
  Reconstructed last so it lands on top.
- Use `useRef` to grab `path.getTotalLength()` after mount; set the
  `--len` CSS variable.

**C2.3 — R-peak markers + playhead + empty/loading/error states**
- R-peak markers per §7.9: clean = outlined triangle (`fill none stroke-clean`),
  reconstructed = filled cyan triangle. Render AFTER traces complete
  (animation-delay matched to trace draw duration), fade-in scale-bounce.
- Hover playhead: track mouse `clientX` in the plot frame; render a vertical
  dashed line + glassmorphism readout in the top-right showing t / clean /
  noisy / recon values at the cursor index (use mono with the per-trace
  color for each value).
- Empty state: when `response === null && status === 'idle'`, overlay
  "AWAITING SIGNAL — CONFIGURE & RUN" centered, with a pulsing 8px dot.
- Loading: pulsing dot + "RUNNING…" overlay.
- Error: magenta callout (NOT red) in the top of the card.

**C2.4 — Legend overlay + scanline + performance check**
- Legend: glassmorphism panel top-right (per §7.9), one line per trace +
  one line per R-peak style. Use `backdrop-filter: blur(6px)` + 80% opacity
  background.
- Add the faint scanline overlay (`::before` with repeating-linear-gradient
  at `rgba(255,255,255,0.012)` per §7.9).
- Performance: subsample to 2500 points client-side if `clean.length > 2500`
  (backend caps but defend in depth). Verify hover playhead is smooth (no
  lag on 2500-point traces).

**Acceptance:** End-to-end against the running backend: pick record 214,
input SNR=0, epochs=10, click Run; within 3s the plot renders three traces
with the draw-on animation, R-peak markers appear after, the legend overlay
shows, hover reveals the playhead readout. Compare side-by-side with
`mockup.html` — colors, stroke widths, and animation timing should match.
No console errors.

---

#### C3 — Metrics row &nbsp; [E]
**Sub-steps:** 1
- `MetricsPanel.tsx` per DESIGN.md §7.11: 5 cells with 1px gap on
  `bg-rule` (the hairline-via-gap trick).
- Per cell: mono small-caps label, huge mono number (28px) with unit in
  `ink-3`, sub-line or progress bar below.
- **Colour rules (DO NOT deviate):** ΔSNR cell uses `accent-cyan` if
  positive, `accent-magenta` if negative. **Never red, never green.**
  rpeak cell has a 3px progress bar (cyan if ≥0.85, magenta if below).
- Numbers count up from previous value via Framer Motion `animate` on a
  state value over 600ms cubic ease-out.
- Show "—" when no response.

---

#### C4 — End-to-end wire-up &nbsp; [E]
**Sub-steps:** 1
- Compose `Layout` with `ControlsPanel` (sidebar), `SignalPlot` (top of main),
  `MetricsPanel` (below plot).
- Implement `runSimulation()` in the store: call `simulate()`, update
  `response` and `status`, push to history (history rendered in D4).

**Acceptance (Milestone C):**
```bash
# Backend running in one shell, frontend in another:
# 1. Open http://localhost:5173
# 2. Select record 214
# 3. Set flicker_freq=50, input_snr=0, epochs=10
# 4. Click Run
# 5. Within ~3 s, plot shows 3 traces with R-peak markers,
#    MetricsPanel shows delta_snr_db > 0 and rpeak_integrity > 0.7
```

This is the v1 minimum-viable demo. Everything beyond is enhancement.

---

### Milestone D — Enhancements

#### D1 — Spike raster &nbsp; [M]
**Sub-steps:** 2. Built per DESIGN.md §7.10 — raw SVG, NO Plotly. Reference
`<svg id="raster-svg">` rendering in `mockup.html`.

**D1.1 — Component**
- `components/SpikeRaster.tsx`: 200px tall SVG canvas sharing the same x-axis
  as `SignalPlot`. Takes `events: { tMs, layer, neuron }[]` and
  `layerSizes: number[]` (default `[32, 16, 8]`).
- Render hairline rules between layers (`stroke-rule stroke-[1] stroke-dasharray="2 4"`).
- Each spike: 2×4px vertical rect, color by layer:
  L1 → `fill-accent-amber`, L2 → `fill-accent-cyan`, L3 → `fill-accent-bone`.
- Layer labels on RIGHT edge in mono small caps (`L1 · INPUT · 32`, etc.).
- Spikes fade in staggered by `t` position (later spikes appear later — the
  raster looks like it's being scanned out). 300ms fade-in with delay
  `600 + (t/duration)*700` ms per event.

**D1.2 — Stack integration**
- Sit BELOW `SignalPlot` inside the SAME bordered card (per DESIGN.md §6
  layout — they share one container with a hairline `border-t border-rule`
  between them so the time-axis alignment is visually obvious).
- Empty/loading/error states match C2.3 of SignalPlot.

---

#### D2 — Export button &nbsp; [E]
**Sub-steps:** 1
- Button in `MetricsPanel` (or header): `<a href={exportCsvUrl(simId)}
  download>Download CSV</a>`. Disabled when no response.

---

#### D3 — File upload UI &nbsp; [M]
**Sub-steps:** 2

**D3.1 — Dropzone**
- `npm i react-dropzone`
- `components/UploadDropzone.tsx`: drag-and-drop area accepting `.csv`.
  Numeric input for `fs`. Posts to `/api/upload`, stores `signal_id` in store.

**D3.2 — Integration**
- Shows up in `ControlsPanel` when source = "Uploaded CSV".
- After successful upload, displays `n_samples` and `fs` summary.
- Error messages from backend (e.g. > 2 columns) surfaced inline.

---

#### D4 — Experiment history &nbsp; [M]
**Sub-steps:** 2

**D4.1 — Store + persistence**
- Define `HistoryEntry = { id, timestamp, params, summaryMetrics }`.
- `addToHistory()` called at end of `runSimulation()`.
- Cap at 20 entries; oldest evicted.

**D4.2 — Panel**
- `components/HistoryPanel.tsx`: list of past runs with timestamp, source,
  ΔSNR, rpeak. Click "Revisit" → restores `params` (does NOT re-run; user
  must hit Run). Click "Delete" removes from history.

---

### Milestone E — Polish

#### E1 — Narrative mode &nbsp; [M]
**Sub-steps:** 2

**E1.1 — Tour library**
- `npm i react-joyride`
- `components/NarrativeTour.tsx`: define 5 steps:
  1. Welcome — what NeuroFlux is, 2 sentences.
  2. Controls panel — pick a record, set noise.
  3. Run button — kick off the simulation.
  4. Signal plot — explain the three traces and R-peak markers.
  5. Passive/trained toggle — the Phase 3 punchline (rpeak=1.000 passive).
- Trigger on first visit (LocalStorage flag) + "Take the tour" button in
  header.

**E1.2 — Copy review**
- Each step has 1–2 sentences. Plain English, no jargon without context.
- Cross-link the Phase 3 doc from step 5.

---

#### E2 — Developer README &nbsp; [E]
**Sub-steps:** 1
- `packages/web/README.md`: one page, four sections:
  - **Prerequisites:** Python 3.11+, Node 20+, `data/mit_bih/` populated.
    Reference (don't duplicate) the §0 setup in `PHASE5_IMPLEMENTATION.md`.
  - **First-time setup on a fresh clone:** the one-liner from §0.3 to
    download the 16 MIT-BIH records (or just `214` for a minimal demo).
  - **Run locally:** two `cd` + two commands (`uv run uvicorn ...` and
    `npm run dev`), exactly the two-shell layout from §0.6.
  - **Smoke test:** what to click, what to expect.

---

#### E3 — Playwright smoke test &nbsp; [M]
**Sub-steps:** 1
- `npm i -D @playwright/test && npx playwright install chromium`
- `tests/e2e/golden_path.spec.ts`: launches both servers via
  `webServer` config, opens `/`, selects record 214, clicks Run, asserts
  `MetricsPanel` shows ΔSNR text and `SignalPlot` SVG appears, within 10 s.
- README documents `npm run test:e2e`.

---

### Milestone F — REMOVED

A WebSocket streaming milestone was previously sketched here for runs > 30 s.
Resolved in §9 (decision #4): 10 s segments are sufficient, so the streaming
path is dropped entirely. The synchronous `POST /api/simulate` endpoint is
the only execution mode in v1, and the `duration_s` field stays capped at
30 s. Do not add WebSocket code, do not add a streaming generator to
`services/simulator.py`, do not add `api/wsClient.ts`. If a future phase
needs > 30 s runs, design a fresh streaming spec then.

---

## 6. Recommended execution order and batches

Batches are guidance; the executor may regroup Easy/Medium items under the
§4 rules.

| Batch | Tasks | Notes |
|---|---|---|
| 1 | A1, A2 | Easy + Easy. Boot the backend. |
| 2 | A3 | Easy on its own — verifies cached records visible. |
| 3 | A4 (M, 2 sub-steps) | Medium alone. The keystone backend endpoint. |
| 4 | A5, A6 | Medium + Easy. Upload + export round out the API. |
| 5 | B1, B2, B3 | Three Easies. Scaffold + style + layout in one go. |
| 6 | B4, B5 | Medium + Medium — combined is fine because surface is tiny. |
| 7 | C1 | Medium alone — the form is the most decision-dense UI work. |
| 8 | **C2 (HARD) — individually** | Sub-steps C2.1 → C2.4 sequential, verify each. |
| 9 | C3, C4 | Easy + Easy. The wire-up moment. |
| 10 | D1 | Medium alone. |
| 11 | D2, D3, D4 | Easy + Medium + Medium. Manageable together. |
| 12 | E1 | Medium alone. |
| 13 | E2, E3 | Easy + Medium. Closes out v1. |

After each batch:
1. Run the per-task acceptance checks.
2. Run **the gate suite**:
   ```bash
   uv run ruff format --check .
   uv run ruff check .
   uv run mypy .
   make test-fast
   cd packages/web/frontend && npm run build
   ```
3. Commit with a conventional-commit message describing the batch
   (`feat(web): <batch summary>`). One commit per batch.
4. Stop and report progress; do not start the next batch without confirmation.

---

## 7. Final verification (after Milestone E)

### Automated
```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
make test-fast                                      # incl. tests/web/
cd packages/web/frontend
npm run build
npm run test:e2e                                    # Playwright smoke
```

### Manual golden-path checklist (must all pass)
- [ ] Open `http://localhost:5173`. Layout renders, no console errors.
- [ ] Records dropdown lists 16 records.
- [ ] Select 214, defaults (50 Hz flicker, 0 dB, 10 ep, τ=10 ms), Run.
  Plot + metrics appear in < 5 s. ΔSNR > 0, rpeak ≥ 0.7.
- [ ] Toggle "Passive" (epochs → 0), Run. rpeak ≈ 1.0, ΔSNR may be ≤ 0 (this
  IS the Phase 3 punchline).
- [ ] Spike raster shows activity in all layers.
- [ ] Download CSV — opens with 4 columns and the expected row count.
- [ ] Upload a 5-row CSV with `fs=250`. Run on it. No crash.
- [ ] History panel lists the last runs. Revisit restores params.
- [ ] Narrative tour completes without console errors.
- [ ] Refresh the page — history persists.
- [ ] Resize the window to 1024 × 768 (tablet) — layout reflows.

### Demo dry-run (literally rehearse before showing evaluators)
Time how long the golden-path takes from page load to "ΔSNR visible". Target
< 60 s including narration. If longer, the demo is too fiddly — simplify the
controls panel.

---

## 8. Things explicitly NOT to do in Phase 5

| Don't | Why |
|---|---|
| Modify `packages/core/` | Engine is frozen. Phase 5 is a UI layer. |
| Modify `packages/datasets/` | Same. If the simulator needs something the
existing API doesn't expose, wrap it in `services/simulator.py`. |
| Skip mypy / ruff on backend | Lifting the `packages/web` exclusion in A1
makes these gates apply. Keep them green. |
| Commit `data/mit_bih/` to git | Stays gitignored. Frontend assumes the
server has it locally. |
| Commit `node_modules/`, `dist/`, `playwright-report/` | Add to `.gitignore`
in B1. |
| Add the baseline-comparison panel | Phase 4 prerequisite. Stub with a
"Coming in Phase 4" message in `MetricsPanel` or omit. |
| Deploy to Vercel/Netlify/GitHub Pages | Out of scope. v1 is local-dev. A
deployment phase comes after Phase 4 produces baselines worth showing. |
| Re-implement the SNN in JS/WASM | See §3.1. |

---

## 9. Resolved decisions (previously open)

All four open questions have been answered. Recorded here so the executor
doesn't reopen them.

| # | Question | Decision | Implication |
|---|---|---|---|
| 1 | Hosting target | **Local-dev only for v1.** Demo is shared via screen-share or tunnel (ngrok/Cloudflare) when needed. | No Dockerfile, no deploy config, no infra files in v1. CORS only needs to allow `localhost:5173`. |
| 2 | Telemetry / analytics | **None.** | Do not add Plausible, Umami, GA, or any tracking script. Frontend ships with zero external network calls beyond the local backend. |
| 3 | Authentication | **None — trusted-setting demo.** | No login UI, no session middleware, no password env var. Backend assumes the network is trusted. |
| 4 | Long runs > 30 s | **No — 10 s segments are enough.** | **Milestone F is dropped entirely.** Synchronous `/simulate` is the only execution mode. `SimulateRequest.duration_s` keeps its `le=30.0` cap. |

---

## 10. Glossary for the executor

- **passive NeuroFlux** = network run with `training_epochs=0`. STDP weights
  never updated; pure membrane filtering + Poisson rate decode.
- **trained NeuroFlux** = `training_epochs >= 1`. STDP shapes weights on the
  flickered signal.
- **rpeak_integrity** = clinical metric in `[0, 1]`. Passive ≈ 1.0; trained
  ≈ 0.5–0.9 depending on epochs.
- **ΔSNR** = output SNR − input SNR (dB). Positive = noise reduced. Negative
  at high input SNR is documented and expected — don't "fix" it.
- **Phase 3 punchline** = passive preserves rpeak perfectly; STDP adds ΔSNR
  but damages rpeak. The dashboard's job is to make this visible.

---

*End of plan.*
