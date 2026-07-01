import asyncio
from fastapi import APIRouter, Request

router = APIRouter(tags=["debug"])


@router.get(
    "/debug/frame-bus/{topic}",
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
            "resolution": f"{frame_ctx.frame_data.shape[1]}x{frame_ctx.frame_data.shape[0]}",
            "channels": frame_ctx.frame_data.shape[2],
            "detections_count": len(detections),
            "detections": [
                {
                    "type": d.detection_type,
                    "label": d.label,
                    "confidence": float(d.confidence),
                    "box": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                }
                for d in detections[:10]
            ],
        }

    except asyncio.TimeoutError:
        return {
            "status": "stale",
            "topic": topic,
            "message": f"No frames received from topic '{topic}' within 3 seconds.",
        }

    finally:
        frame_bus.unsubscribe(topic, queue)


@router.get(
    "/debug/frame-bus-topics",
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