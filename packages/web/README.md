# NeuroFlux Web Dashboard — Developer Guide

This package contains the interactive web dashboard for NeuroFlux. It consists of a **FastAPI backend** that wraps the Python SNN engine and a **React frontend** built with Vite and Tailwind CSS.

## Prerequisites

- **Python 3.11** (Required by Numba/SNN engine)
- **Node.js 20+**
- **MIT-BIH Data**: The dashboard requires record 214 to be downloaded locally.

## First-time Setup

From the repository root:

1.  **Install Python dependencies:**
    ```bash
    uv sync --all-packages
    ```

2.  **Download demo records:**
    ```bash
    uv run python -c "from neuroflux_datasets.mit_bih.downloader import download_record; download_record('214', 'data/mit_bih')"
    ```

3.  **Install Frontend dependencies:**
    ```bash
    cd packages/web/frontend && npm install
    ```

## Running Locally

You will need two terminals running from the repository root:

**Terminal 1 (Backend):**
```bash
export PATH=$PATH:~/.local/bin # if uv is not in path
cd packages/web/backend
uv run uvicorn neuroflux_web.app:create_app --factory --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd packages/web/frontend
npm run dev
```

The dashboard will be available at [http://localhost:5173](http://localhost:5173).

## Smoke Test (Golden Path)

1.  Ensure "RECORD 214" is selected in the Source panel.
2.  Click **"Run Simulation"**.
3.  Observe the **Hero Trace** drawing clean, noisy, and reconstructed signals.
4.  Verify **R-Peak Integrity** markers appear as triangles.
5.  Toggle **"Passive Mode"** and re-run to see perfect R-peak preservation at the cost of SNR.

## Technical Architecture

- **Engine Fidelity:** The dashboard uses the exact same `neuroflux_core` package as the research scripts, ensuring numerical parity with published results.
- **JIT Warmup:** On startup, the backend runs a 1ms dummy simulation to pre-compile Numba kernels, preventing UI lag on the first user request.
- **Design System:** Follows the "Prussian Blueprint" aesthetic documented in `packages/web/design/DESIGN.md`.
