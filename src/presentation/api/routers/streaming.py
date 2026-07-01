from typing import List, Optional
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from domain.models.camera_configuration import CameraConfiguration
from presentation.live_stream_handler import LiveStreamHandler

router = APIRouter(tags=["stream"])


def get_settings(request: Request):
    return request.app.state.settings



def get_audio_overlay_state(request: Request):
    audio_overlay_state = getattr(request.app.state, "audio_overlay_state", None)

    if audio_overlay_state is not None:
        return audio_overlay_state

    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is not None:
        return getattr(orchestrator, "audio_overlay_state", None)

    return None


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

    if not config.camera.active:
        raise HTTPException(status_code=400, detail="Camera is not active")

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

    topic = f"{camera_id}_ai" if active_detections else camera_id

    handler = LiveStreamHandler(
        topic=topic,
        frame_bus=frame_bus,
        detector_provider=get_detector_provider(request),
        visible_types=active_detections,
        camera_config=config,
        audio_overlay_state=get_audio_overlay_state(request),
    )

    return StreamingResponse(
        handler.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get(
    "/view",
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
                    max-width: 1600px;
                    margin: 0 auto;
                }}
                .stream {{
                    max-width: 100%;
                    width: 100%;
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