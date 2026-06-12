import asyncio
import io
from typing import List, Optional
from urllib.parse import urlencode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from analytics.analyze_chickens import ChickenDataAnalyzer
from domain.camera_configuration import CameraConfiguration
from streaming.mjpeg_stream_handler import LiveStreamHandler


router = APIRouter()


def get_settings(request: Request):
    return request.app.state.settings


def get_detector_provider(request: Request):
    detector_provider = getattr(request.app.state, "detector_provider", None)
    if detector_provider is not None:
        return detector_provider

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        detection_manager = getattr(orchestrator, "detection_manager", None)
        if detection_manager is not None:
            return getattr(detection_manager, "detector_provider", None)

    return None


def parse_stream_request(
    cam: str = Query(
        ...,
        description="Camera ID from nvr_registry.yaml, e.g. h8c.",
        examples=["h8c"],
    ),
    detect: Optional[List[str]] = Query(
        None,
        description=(
            "Detection types. Use repeated query params, e.g. "
            "?detect=chicken_detection&detect=person_detection."
        ),
        examples=[["chicken_detection"], ["person_detection"]],
    ),
    settings=Depends(get_settings),
):
    config: CameraConfiguration = settings.registry.get(cam)

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Camera with ID {cam} is missing from the NVR registry.",
        )

    if not config.active:
        raise HTTPException(
            status_code=400,
            detail="The requested camera pipeline is currently disabled.",
        )

    requested_detections = detect or []

    allowed_detections = [
        d.detection_type
        for d in settings.detection_configs.get(cam, [])
    ]

    for requested in requested_detections:
        if requested not in allowed_detections:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Detection type '{requested}' is not configured "
                    f"for camera {cam}. Configured options: "
                    f"{allowed_detections}"
                ),
            )

    return {
        "config": config,
        "requested_detections": requested_detections,
    }


@router.get(
    "/stream",
    tags=["stream"],
    summary="MJPEG stream from raw or AI camera topic",
)
async def get_live_stream(
    request: Request,
    params: dict = Depends(parse_stream_request),
):
    config: CameraConfiguration = params["config"]
    active_detections: List[str] = params["requested_detections"]

    frame_bus = request.app.state.frame_bus
    camera_id = config.camera.id

    # Only this router decides the topic. LiveStreamHandler must not append _ai.
    topic = f"{camera_id}_ai" if active_detections else camera_id

    handler = LiveStreamHandler(
        topic=topic,
        frame_bus=frame_bus,
        detector_provider=get_detector_provider(request),
        visible_types=active_detections,
    )

    return StreamingResponse(
        handler.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get(
    "/view",
    tags=["stream"],
    summary="HTML camera preview page",
    response_class=HTMLResponse,
)
def view_camera_page(
    cam: str = Query(..., examples=["h8c"]),
    detect: Optional[List[str]] = Query(
        None,
        examples=[["chicken_detection"], ["person_detection"]],
    ),
):
    query = [("cam", cam)]

    for detection_type in detect or []:
        query.append(("detect", detection_type))

    stream_url = "/stream?" + urlencode(query)

    html_content = f"""
    <html>
        <head>
            <title>Podglad Kamery {cam}</title>
            <style>
                body {{
                    background: #111;
                    color: white;
                    text-align: center;
                    font-family: sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .stream {{
                    max-width: 90%;
                    border: 4px solid #333;
                    margin-top: 20px;
                    border-radius: 8px;
                }}
                h1 {{ color: #ffa500; }}
                code {{ color: #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>System CCTV - Kamera: {cam}</h1>
                <div><code>{stream_url}</code></div>
                <img class="stream" src="{stream_url}" />
            </div>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get(
    "/analytics/plot",
    tags=["analytics"],
    summary="Analytics dashboard as PNG",
)
async def get_analytics_plot(cam: str, detect: str):
    csv_path = f"archive_logs/{cam}_{detect}.csv"
    analyzer = ChickenDataAnalyzer(csv_path)

    data_loaded = await asyncio.to_thread(analyzer.load_data)
    if not data_loaded:
        raise HTTPException(
            status_code=404,
            detail="Brak danych do wygenerowania wykresu.",
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    analyzer.plot_activity_timeline(ax1)
    analyzer.plot_spatial_heatmap(ax2)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)
    plt.close(fig)

    return StreamingResponse(img_buf, media_type="image/png")


@router.get(
    "/debug/frame-bus/{topic}",
    tags=["debug"],
    summary="Read one frame from a frame_bus topic",
)
async def debug_frame_bus(topic: str, request: Request):
    frame_bus = request.app.state.frame_bus

    queue = frame_bus.subscribe(topic)

    try:
        frame_ctx = await asyncio.wait_for(queue.get(), timeout=3.0)
        detections = getattr(frame_ctx, "detections", []) or []

        return {
            "status": "active",
            "topic": topic,
            "frame_camera_id": frame_ctx.camera_id,
            "frame_id": frame_ctx.frame_id,
            "frame_uid": getattr(frame_ctx, "frame_uid", None),
            "stream_session_id": getattr(frame_ctx, "stream_session_id", None),
            "scene_id": getattr(frame_ctx, "scene_id", None),
            "resolution": (
                f"{frame_ctx.frame_data.shape[1]}x"
                f"{frame_ctx.frame_data.shape[0]}"
            ),
            "channels": frame_ctx.frame_data.shape[2],
            "detections_count": len(detections),
            "detections": [
                {
                    "type": d.detection_type,
                    "label": d.label,
                    "confidence": float(d.confidence),
                    "box": [
                        int(d.x1),
                        int(d.y1),
                        int(d.x2),
                        int(d.y2),
                    ],
                }
                for d in detections[:10]
            ],
        }

    except asyncio.TimeoutError:
        return {
            "status": "stale",
            "topic": topic,
            "message": (
                f"No frames received from topic '{topic}' "
                f"within 3 seconds."
            ),
        }

    finally:
        frame_bus.unsubscribe(topic, queue)


@router.get(
    "/debug/frame-bus-topics",
    tags=["debug"],
    summary="List frame_bus topics and subscriber counts",
)
async def debug_frame_bus_topics(request: Request):
    frame_bus = request.app.state.frame_bus
    subscribers = getattr(frame_bus, "_subscribers", {})

    return {
        "topics": {
            topic: len(queues)
            for topic, queues in subscribers.items()
        }
    }
