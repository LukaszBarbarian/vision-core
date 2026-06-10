import asyncio
import io
from fastapi import APIRouter, Query, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import List, Optional
from analytics.analyze_chickens import ChickenDataAnalyzer
from core.settings import Settings
from domain.camera_configuration import CameraConfiguration
from streaming.mjpeg_stream_handler import LiveStreamHandler

# 📊 NOWOŚĆ: Importy potrzebne do obsługi wykresów w pamięci RAM i bez okienek GUI
import matplotlib
matplotlib.use('Agg')  # Wyłącza szukanie ekranu przez Matplotlib (serwer-friendly)
import matplotlib.pyplot as plt


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

    if not active_detections:
        handler = LiveStreamHandler(camera_id, frame_bus)
        return StreamingResponse(
            handler.generate_mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    ai_camera_topic = f"{camera_id}_ai"
    handler = LiveStreamHandler(ai_camera_topic, frame_bus)

    return StreamingResponse(
        handler.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/view")
def view_camera_page(
    cam: str,
    detect: Optional[List[str]] = Query(None)
):
    # Generujemy końcówkę URL dla tagu <img>, np. "&detect=chicken_detection"
    detect_params = "".join([f"&detect={d}" for d in detect]) if detect else ""

    html_content = f"""
    <html>
        <head>
            <title>Podgląd Kamery {cam}</title>
            <style>
                body {{ background: #111; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stream {{ max-width: 90%; border: 4px solid #333; margin-top: 20px; border-radius: 8px; }}
                h1 {{ color: #ffa500; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>System CCTV - Kamera: {cam}</h1>
                <img class="stream" src="/stream?cam={cam}{detect_params}" />
            </div>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


# 📊 Czyste API (zostawione w tle, gdybyś chciał sprawdzić stan w przeglądarce pod osobnym URL)
@router.get("/analytics/plot")
async def get_analytics_plot(cam: str, detect: str):
    """Generuje wykresy w pamięci RAM i zwraca je jako surowy obraz PNG."""
    csv_path = f"archive_logs/{cam}_{detect}.csv"
    analyzer = ChickenDataAnalyzer(csv_path)
    
    data_loaded = await asyncio.to_thread(analyzer.load_data)
    if not data_loaded:
        raise HTTPException(status_code=404, detail="Brak danych do wygenerowania wykresu.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    analyzer.plot_activity_timeline(ax1)
    analyzer.plot_spatial_heatmap(ax2)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close(fig)

    return StreamingResponse(img_buf, media_type="image/png")


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