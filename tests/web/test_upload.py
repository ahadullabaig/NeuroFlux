import pytest
from httpx import AsyncClient, ASGITransport
from neuroflux_web.app import create_app
import io

@pytest.mark.asyncio
async def test_upload_endpoint():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Happy path
        csv_content = "value\n1.0\n2.0\n3.0\n4.0\n5.0"
        files = {"file": ("test.csv", csv_content, "text/csv")}
        data = {"fs": "250"}
        
        response = await ac.post("/api/upload", files=files, data=data)
        assert response.status_code == 200
        res_data = response.json()
        assert "signal_id" in res_data
        assert res_data["n_samples"] == 5
        
        # Test rejection of multi-column
        csv_bad = "v1,v2,v3\n1,2,3"
        files_bad = {"file": ("bad.csv", csv_bad, "text/csv")}
        response = await ac.post("/api/upload", files=files_bad, data=data)
        assert response.status_code == 400
