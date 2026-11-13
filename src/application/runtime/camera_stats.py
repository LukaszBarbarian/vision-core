"""
Live per-camera health: is the stream delivering frames, at what rate and
resolution, and how long the detectors take. Read by the dashboard and the
stream overlay.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

STALLED_AFTER_SECONDS = 5.0
FPS_WINDOW_SECONDS = 5.0


@dataclass
class CameraStats:
    camera_id: str
    connected: bool = False
    width: int | None = None
    height: int | None = None
    last_frame_at: float | None = None
    last_error: str | None = None
    inference_ms: dict[str, float] = field(default_factory=dict)
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=300))

    def fps(self, now: float) -> float:
        recent = [t for t in self._frame_times if now - t <= FPS_WINDOW_SECONDS]

        if len(recent) < 2:
            return 0.0

        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def status(self, now: float) -> str:
        if self.last_frame_at is not None and now - self.last_frame_at <= STALLED_AFTER_SECONDS:
            return "live"

        if self.connected:
            return "stalled"

        return "offline"


class CameraStatsRegistry:
    def __init__(self):
        self._cameras: dict[str, CameraStats] = {}

    def get(self, camera_id: str) -> CameraStats:
        stats = self._cameras.get(camera_id)

        if stats is None:
            stats = self._cameras[camera_id] = CameraStats(camera_id=camera_id)

        return stats

    def record_connection(self, camera_id: str, connected: bool, error: str | None = None) -> None:
        stats = self.get(camera_id)
        stats.connected = connected

        if error is not None:
            stats.last_error = error

    def record_frame(self, camera_id: str, width: int, height: int, timestamp: float | None = None) -> None:
        stats = self.get(camera_id)
        now = timestamp if timestamp is not None else time.time()

        stats.width, stats.height = int(width), int(height)
        stats.last_frame_at = now
        stats.connected = True
        stats._frame_times.append(now)

    def record_inference(self, camera_id: str, detection_type: str, milliseconds: float) -> None:
        self.get(camera_id).inference_ms[detection_type] = float(milliseconds)

    def snapshot(self, camera_id: str, now: float | None = None) -> dict:
        now = now if now is not None else time.time()
        stats = self.get(camera_id)

        return {
            "camera_id": camera_id,
            "status": stats.status(now),
            "fps": round(stats.fps(now), 1),
            "resolution": f"{stats.width}×{stats.height}" if stats.width else None,
            "last_frame_age_s": round(now - stats.last_frame_at, 1) if stats.last_frame_at else None,
            "inference_ms": {k: round(v) for k, v in sorted(stats.inference_ms.items())},
            "last_error": stats.last_error,
        }
