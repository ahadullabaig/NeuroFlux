import os
import json
from fastapi import APIRouter, HTTPException, Path, Query
from typing import List, Optional
from neuroflux_datasets.mit_bih.annotator import extract_clean_segments
import numpy as np

router = APIRouter()

# Helper to find where manifest and data live
# Use absolute path to avoid issues with current working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
# routes/ is 1, neuroflux_web/ is 2, src/ is 3, backend/ is 4, web/ is 5, packages/ is 6
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "..", ".."))
MANIFEST_PATH = os.path.join(_ROOT, "packages", "datasets", "src", "neuroflux_datasets", "mit_bih", "manifest.json")
DATA_DIR = os.path.join(_ROOT, "data", "mit_bih")

@router.get("/records")
async def list_records():
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=500, detail="Manifest not found")
    
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    
    results = []
    for record_id, meta in manifest["records"].items():
        # Check if actually cached (dat file exists)
        dat_path = os.path.join(DATA_DIR, f"{record_id}.dat")
        if os.path.exists(dat_path):
            # For v1, we assume 250Hz and 30min duration as per MIT-BIH defaults
            # In a more robust version, we'd read this from wfdb headers
            results.append({
                "record_id": record_id,
                "fs": 250.0,
                "duration_s": 1800.0,
                "channels": ["MLII", "V1"] # MIT-BIH defaults
            })
    
    return results

@router.get("/records/{id}/segment")
async def get_record_segment(
    id: str = Path(..., description="Record ID"),
    duration_s: float = Query(10.0, gt=0, le=30.0)
):
    try:
        # extract_clean_segments(record_id, cache_dir, duration_s, fs=250.0, *, channel=0)
        segments = extract_clean_segments(id, DATA_DIR, duration_s=duration_s)
        if not segments:
            raise HTTPException(status_code=404, detail=f"No clean segments found for record {id}")
        
        # Take the first one
        segment = segments[0]
        
        # segment is a 1D float64 array as per annotator.py
        return {
            "record_id": id,
            "fs": 250.0, # Target fs is 250.0 by default in extract_clean_segments
            "signal": segment.tolist()
        }
    except Exception as e:
        if "No clean segments" in str(e):
             raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Error extracting segment: {str(e)}")
