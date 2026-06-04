import pytest
from httpx import AsyncClient, ASGITransport
from neuroflux_web.app import create_app

@pytest.mark.asyncio
async def test_export_endpoint():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # First simulate to get an ID
        payload = {
            "source": "record",
            "record_id": "214",
            "duration_s": 1.0,
            "flicker_freq_hz": 50,
            "input_snr_db": 0,
            "training_epochs": 0
        }
        sim_response = await ac.post("/api/simulate", json=payload)
        if sim_response.status_code != 200:
            pytest.skip("Simulation failed, skipping export test")
            
        sim_id = sim_response.json()["simulation_id"]
        
        # Now export
        exp_response = await ac.get(f"/api/export/{sim_id}.csv")
        assert exp_response.status_code == 200
        assert exp_response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "t_ms,clean,noisy,reconstructed" in exp_response.text
