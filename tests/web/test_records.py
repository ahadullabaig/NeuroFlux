import pytest
from httpx import AsyncClient, ASGITransport
from neuroflux_web.app import create_app
import os

@pytest.mark.asyncio
async def test_records_endpoints():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Test listing
        response = await ac.get("/api/records")
        assert response.status_code == 200
        data = response.json()
        
        # If we downloaded 214, it should be in the list
        record_ids = [r["record_id"] for r in data]
        if "214" in record_ids:
            # Test segment extraction
            seg_response = await ac.get("/api/records/214/segment?duration_s=10")
            assert seg_response.status_code == 200
            seg_data = seg_response.json()
            assert seg_data["record_id"] == "214"
            assert "signal" in seg_data
            assert len(seg_data["signal"]) > 0
        else:
            pytest.skip("Record 214 not found in data/mit_bih, skipping segment test")
