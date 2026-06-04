from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from neuroflux_web.routes.simulate import simulation_cache
import io
import csv

router = APIRouter()

@router.get("/export/{simulation_id}.csv")
async def export_simulation(simulation_id: str):
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found or expired")
    
    sim = simulation_cache[simulation_id]
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["t_ms", "clean", "noisy", "reconstructed"])
    
    for i in range(len(sim.t_ms)):
        writer.writerow([
            sim.t_ms[i],
            sim.clean[i],
            sim.noisy[i],
            sim.reconstructed[i]
        ])
    
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=neuroflux_export_{simulation_id}.csv"}
    )
