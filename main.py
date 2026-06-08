import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routers.stream_router import router as stream_router
from core.orchestrator import AppOrchestrator
from core.settings import Settings





@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    orchestrator = AppOrchestrator(settings=settings)

    app.state.orchestrator = orchestrator
    app.state.frame_bus = orchestrator.frame_bus
    app.state.settings = settings

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

app.include_router(stream_router)


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
