import logging
import ssl
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from application.orchestration.app_orchestrator import AppOrchestrator
from infrastructure.config.settings import Settings
from presentation.api.routers.analytics import router as analytics_router
from presentation.api.routers.streaming import router as cameras_router

ssl._create_default_https_context = ssl._create_unverified_context


class _SkipPolledEndpoints(logging.Filter):
    """The dashboard polls /status.json every 2 s; keep it out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/status.json" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_SkipPolledEndpoints())


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    orchestrator = AppOrchestrator(settings=settings)

    app.state.orchestrator = orchestrator
    app.state.settings = settings

    app.state.frame_bus = orchestrator.frame_bus
    app.state.audio_bus = orchestrator.audio_bus
    app.state.event_bus = orchestrator.event_bus

    app.state.audio_overlay_state = orchestrator.audio_overlay_state
    app.state.camera_stats = orchestrator.camera_stats
    app.state.face_identity_store = orchestrator.face_identity_store

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
        "status": "ok",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False,
    )