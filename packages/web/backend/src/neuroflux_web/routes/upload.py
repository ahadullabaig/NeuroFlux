import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from cachetools import TTLCache
import numpy as np
import io
import csv

router = APIRouter()

# In-memory storage for uploaded signals
# TTL 1 hour, max 50 entries
UPLOAD_STORE = TTLCache(maxsize=50, ttl=3600)

@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    fs: float = Form(...)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        reader = csv.reader(io.StringIO(decoded))
        
        # Read all rows, skipping header if necessary
        data = []
        for i, row in enumerate(reader):
            if not row:
                continue
            
            # Simple header detection: if first row cells are not numeric
            if i == 0:
                try:
                    float(row[0])
                except ValueError:
                    continue # It's a header
            
            if len(row) > 2:
                raise HTTPException(status_code=400, detail="CSV must have 1 or 2 columns maximum")
            
            try:
                # We take the first column as the signal
                data.append(float(row[0]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Non-numeric value at row {i+1}")
        
        if not data:
            raise HTTPException(status_code=400, detail="CSV is empty")
        
        if len(data) > 100000:
             raise HTTPException(status_code=400, detail="CSV too large (max 100k samples)")

        signal_id = uuid.uuid4().hex
        signal_array = np.array(data, dtype=np.float64)
        UPLOAD_STORE[signal_id] = (fs, signal_array)
        
        return {
            "signal_id": signal_id,
            "fs": fs,
            "n_samples": len(data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
