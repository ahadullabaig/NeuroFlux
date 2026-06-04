import pytest
from httpx import AsyncClient, ASGITransport
from neuroflux_web.app import create_app

@pytest.mark.asyncio
async def test_simulate_endpoint():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # We need record 214 downloaded
        payload = {
            "source": "record",
            "record_id": "214",
            "duration_s": 2.0, # Short duration for faster test
            "flicker_freq_hz": 50,
            "input_snr_db": 0,
            "training_epochs": 1
        }
        response = await ac.post("/api/simulate", json=payload)
        
        # Check if record 214 is missing (skip if so)
        if response.status_code == 404 and "No clean segments" in response.text:
            pytest.skip("Record 214 not found or no clean segments")
            
        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert "metrics" in data
        assert data["metrics"]["delta_snr_db"] is not None
        assert "spike_events" in data
        assert len(data["spike_events"]) > 0
