import pytest
from httpx import AsyncClient, ASGITransport
from neuroflux_web.app import create_app

@pytest.mark.asyncio
async def test_health_endpoint():
    app = create_app()
    # Manually trigger startup to run warmup
    # FastAPI test client with lifespan
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Wait a bit for startup if needed, but AsyncClient with app handles lifespan
        response = await ac.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        # Since we are running in tests, and startup_event is async, 
        # it should have completed by the time we make the request if handled correctly.
        assert "numba_warm" in data
