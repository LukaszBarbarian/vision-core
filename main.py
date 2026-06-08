import uvicorn
from fastapi import FastAPI
from src.api.api_router import router as stream_router

app = FastAPI(
    title="Vision-Core Automated Streaming Platform", 
    version="1.0.0"
)

# Attach routes
app.include_router(stream_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)