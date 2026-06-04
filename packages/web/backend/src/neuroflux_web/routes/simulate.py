import asyncio
from fastapi import APIRouter, HTTPException
from cachetools import TTLCache
from neuroflux_web.models import SimulateRequest, SimulateResponse
from neuroflux_web.services.simulator import run_simulation
from neuroflux_web.routes.records import DATA_DIR
from neuroflux_datasets.mit_bih.annotator import extract_clean_segments
import numpy as np

router = APIRouter()

# Global lock to serialize simulations (engine is CPU-heavy and may have JIT lock)
SIMULATION_LOCK = asyncio.Lock()

# LRU Cache for simulation results (to support /api/export)
# TTL 1 hour, max 20 entries as per spec
simulation_cache = TTLCache(maxsize=20, ttl=3600)

# Reference to upload store (implemented in upload.py)
# For now, we'll use a global dict shared between them
from neuroflux_web.routes.upload import UPLOAD_STORE

@router.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    async with SIMULATION_LOCK:
        try:
            # 1. Load source signal
            if req.source == "record":
                if not req.record_id:
                    raise HTTPException(status_code=400, detail="record_id required for source='record'")
                
                # Use same logic as records.py but we need the raw signal
                segments = extract_clean_segments(req.record_id, DATA_DIR, duration_s=req.duration_s)
                if not segments:
                    raise HTTPException(status_code=404, detail=f"No clean segments for {req.record_id}")
                clean_signal = segments[0]
                fs = 250.0
                
            elif req.source == "upload":
                if not req.signal_id:
                    raise HTTPException(status_code=400, detail="signal_id required for source='upload'")
                
                if req.signal_id not in UPLOAD_STORE:
                    raise HTTPException(status_code=404, detail="Uploaded signal not found or expired")
                
                fs, clean_signal = UPLOAD_STORE[req.signal_id]
                # If duration_s is provided, we might need to truncate/pad, 
                # but spec says duration_s is for simulation.
                # Usually we'd slice the signal to duration_s.
                n_samples = int(req.duration_s * fs)
                if len(clean_signal) > n_samples:
                    clean_signal = clean_signal[:n_samples]
                elif len(clean_signal) < n_samples:
                    # Pad with last value or zeros
                    clean_signal = np.pad(clean_signal, (0, n_samples - len(clean_signal)), 'edge')
            
            else:
                raise HTTPException(status_code=400, detail="Invalid source")

            # 2. Run simulation
            # This is CPU bound, so we run in a thread to not block event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_simulation, req, clean_signal, fs)
            
            # 3. Cache result for export
            simulation_cache[response.simulation_id] = response
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
