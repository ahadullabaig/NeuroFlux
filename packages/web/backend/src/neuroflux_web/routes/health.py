from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "numba_warm": getattr(request.app.state, "numba_warm", False)
    }
