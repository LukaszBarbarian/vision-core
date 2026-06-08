import asyncio
from fastapi import APIRouter, Query, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from typing import List, Optional
from core.settings import Settings
from domain.camera_configuration import CameraConfiguration
from streaming.mjpeg_stream_handler import LiveStreamHandler

router = APIRouter()

def get_settings(request: Request):
    return request.app.state.settings

def parse_stream_request(
    cam: str,
    detect: Optional[List[str]] = Query(None),
    settings = Depends(get_settings) 
):
    config: CameraConfiguration = settings.registry.get(cam)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Camera with ID {cam} is missing from the NVR registry."
        )

    if not config.active:
        raise HTTPException(
            status_code=400,
            detail="The requested camera pipeline is currently disabled."
        )

    requested_detections = detect or []

    # ✔ FIX: źródło prawdy
    allowed_detections = [
        d.detection_type
        for d in settings.detection_configs.get(cam, [])
    ]

    for rd in requested_detections:
        if rd not in allowed_detections:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Detection type '{rd}' is not configured for camera {cam}. "
                    f"Configured options: {allowed_detections}"
                )
            )

    return {
        "config": config,
        "requested_detections": requested_detections
    }



@router.get("/stream")
async def get_live_stream(
    request: Request,
    params: dict = Depends(parse_stream_request)
):
    config: CameraConfiguration = params["config"]
    active_detections: List[str] = params["requested_detections"]
    frame_bus = request.app.state.frame_bus
    camera_id = config.camera.id

    # 1. Czysty stream (bez AI)
    if not active_detections:
        handler = LiveStreamHandler(camera_id, frame_bus)
        return StreamingResponse(
            handler.generate_mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # 2. IMPLEMENTACJA AI: Jeśli użytkownik zażądał detekcji,
    # LiveStreamHandler zaczyna słuchać szyny z dopiskiem _ai, 
    # na którą nasz DetectionWorker właśnie wypycha klatki.
    ai_camera_topic = f"{camera_id}_ai"
    handler = LiveStreamHandler(ai_camera_topic, frame_bus)

    return StreamingResponse(
        handler.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/view")
def view_camera_page(
    cam: str,
    detect: Optional[List[str]] = Query(None) # NOWOŚĆ: Przechwytujemy detekcje w widoku HTML
):
    # Generujemy końcówkę URL dla tagu <img>, np. "&detect=car_detection"
    detect_params = "".join([f"&detect={d}" for d in detect]) if detect else ""

    html_content = f"""
    <html>
        <head>
            <title>Podgląd Kamery {cam}</title>
            <style>
                body {{ background: #111; color: white; text-align: center; font-family: sans-serif; }}
                img {{ max-width: 90%; border: 4px solid #333; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>System CCTV - Kamera: {cam}</h1>
            <img src="/stream?cam={cam}{detect_params}" />
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)






@router.get("/debug/frame-bus/{camera_id}")
async def debug_frame_bus(camera_id: str, request: Request):

    frame_bus = request.app.state.frame_bus

    try:
        queue = frame_bus.subscribe(camera_id)
        frame_ctx = await asyncio.wait_for(queue.get(), timeout=3.0)

        return {
            "status": "active",
            "camera_id": frame_ctx.camera_id,
            "frame_id": frame_ctx.frame_id,
            "resolution": f"{frame_ctx.frame_data.shape[1]}x{frame_ctx.frame_data.shape[0]}",
            "channels": frame_ctx.frame_data.shape[2]
        }

    except asyncio.TimeoutError:
        return {
            "status": "stale",
            "message": f"No frames received from camera '{camera_id}' within 3 seconds."
        }
