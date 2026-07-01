# application/runtime/audio_overlay_state.py

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import threading

import numpy as np


@dataclass(slots=True)
class AudioOverlayInfo:
    camera_id: str
    rms: float = 0.0
    dbfs: float = -120.0
    has_voice: bool = False
    last_text: str = ""
    last_text_at: float = 0.0
    updated_at: float = 0.0


class AudioOverlayState:
    def __init__(
        self,
        voice_dbfs_threshold: float = -45.0,
        stale_after_seconds: float = 3.0,
    ):
        self.voice_dbfs_threshold = float(voice_dbfs_threshold)
        self.stale_after_seconds = float(stale_after_seconds)

        self._lock = threading.Lock()
        self._by_camera: dict[str, AudioOverlayInfo] = {}

    def update_audio_chunk(self, audio_ctx) -> None:
        camera_id = audio_ctx.camera_id
        pcm = np.asarray(audio_ctx.pcm, dtype=np.int16).reshape(-1)

        if pcm.size == 0:
            return

        audio_float = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio_float))))

        dbfs = 20.0 * math.log10(max(rms, 1e-8))
        has_voice = dbfs >= self.voice_dbfs_threshold

        now = time.time()

        with self._lock:
            info = self._by_camera.get(camera_id)

            if info is None:
                info = AudioOverlayInfo(camera_id=camera_id)
                self._by_camera[camera_id] = info

            info.rms = rms
            info.dbfs = dbfs
            info.has_voice = has_voice
            info.updated_at = now

    def update_transcript(self, event: dict) -> None:
        camera_id = event.get("camera_id")
        text = str(event.get("text") or "").strip()

        if not camera_id or not text:
            return

        now = time.time()

        with self._lock:
            info = self._by_camera.get(camera_id)

            if info is None:
                info = AudioOverlayInfo(camera_id=camera_id)
                self._by_camera[camera_id] = info

            info.last_text = text
            info.last_text_at = now
            info.updated_at = now

    def get(self, camera_id: str) -> AudioOverlayInfo | None:
        with self._lock:
            info = self._by_camera.get(camera_id)

            if info is None:
                return None

            return AudioOverlayInfo(
                camera_id=info.camera_id,
                rms=info.rms,
                dbfs=info.dbfs,
                has_voice=info.has_voice,
                last_text=info.last_text,
                last_text_at=info.last_text_at,
                updated_at=info.updated_at,
            )