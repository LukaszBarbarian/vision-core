from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
from src.config.settings import settings
from src.config.models.detection import CameraConfiguration
from src.streaming.stream_handler import LiveStreamHandler

router = APIRouter()

def parse_stream_request(cam: str, detect: Optional[List[str]] = Query(None)):
    config: Optional[CameraConfiguration] = settings.registry.get(cam)
    if not config:
        raise HTTPException(status_code=404, detail=f"Camera with ID {cam} is missing from the NVR registry.")
    
    if not config.active:
        raise HTTPException(status_code=400, detail="The requested camera pipeline is currently disabled.")
        
    requested_detections = detect if detect else []
    allowed_detections = [d.detection_type for d in config.detections]
    for rd in requested_detections:
        if rd not in allowed_detections:
            raise HTTPException(
                status_code=400, 
                detail=f"Detection type '{rd}' is not configured for camera {cam}. Configured options: {allowed_detections}"
            )
            
    return {
        "config": config,
        "requested_detections": requested_detections
    }

@router.get("/stream")
def get_live_stream(params: dict = Depends(parse_stream_request)):
    """
    Stream orchestrator endpoint. Dispatches raw stream or AI-overlay path.
    """
    config: CameraConfiguration = params["config"]
    active_detections: List[str] = params["requested_detections"]
    
    camera_url = config.camera.url
    
    if not active_detections:
        # Scenario A: Pass-through raw live video feed using the stream handler
        handler = LiveStreamHandler(camera_url)
        return StreamingResponse(
            handler.generate_mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        # Scenario B: Live video feed combined with active AI overlay (handled in next steps)
        return {
            "status": "AI overlay streaming path not implemented yet",
            "camera_name": config.camera.name,
            "active_analytics": active_detections
        }