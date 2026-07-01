import time
from typing import Any


class FrameContext:
    def __init__(
        self,
        camera_id: str,
        frame_id: int,
        frame_data: Any,
        detections: list | None = None,
        timestamp: float | None = None,
        stream_session_id: str | None = None,
        scene_id: str | None = None,
    ):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.frame_data = frame_data
        self.detections = detections if detections is not None else []
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.stream_session_id = stream_session_id
        self.scene_id = scene_id

    @property
    def frame_uid(self) -> str:
        session = self.stream_session_id or "no-session"
        return f"{self.camera_id}:{session}:{self.frame_id}"

    @property
    def is_empty(self) -> bool:
        return (
            self.frame_data is None
            or getattr(self.frame_data, "size", 0) == 0
        )
