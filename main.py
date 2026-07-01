import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI
from application.orchestration.app_orchestrator import AppOrchestrator
from infrastructure.config.settings import Settings
from presentation.api.routers.analytics import router as analytics_router
from presentation.api.routers.streaming import router as cameras_router

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    orchestrator = AppOrchestrator(settings=settings)

    app.state.orchestrator = orchestrator
    app.state.frame_bus = orchestrator.frame_bus
    app.state.settings = settings
    app.state.detector_provider = orchestrator.detection_manager.detector_provider

    print("[SYSTEM] Starting Vision Core...")

    await orchestrator.start()

    try:
        yield
    finally:
        print("[SYSTEM] Stopping Vision Core...")
        await orchestrator.stop()


app = FastAPI(
    title="Vision Core",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(analytics_router)
app.include_router(cameras_router)


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
