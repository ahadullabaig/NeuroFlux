from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neuroflux_web.routes import health, records, simulate, upload, export
import numpy as np
from neuroflux_core.network.feedforward import FeedForwardNetwork
from neuroflux_core.network.simulator import Simulator

def create_app() -> FastAPI:
    app = FastAPI(title="NeuroFlux API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(records.router, prefix="/api")
    app.include_router(simulate.router, prefix="/api")
    app.include_router(upload.router, prefix="/api")
    app.include_router(export.router, prefix="/api")

    app.state.numba_warm = False

    @app.on_event("startup")
    async def startup_event():
        # Numba warmup: run a tiny 1ms simulation to JIT-compile kernels
        try:
            # Using FeedForwardNetwork.build for convenience
            network = FeedForwardNetwork.build(layers=[8, 8], seed=0)
            sim = Simulator(network, dt=1.0, seed=0)
            # 1ms simulation
            sim.run(np.zeros((1, 8)), duration_ms=1.0)
            app.state.numba_warm = True
        except Exception as e:
            print(f"Warmup failed: {e}")

    return app
