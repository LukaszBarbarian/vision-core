# infrastructure/speech/speech_worker.py

from __future__ import annotations

import asyncio
import time
import traceback
from collections import deque
from typing import Deque

import numpy as np

from domain.models.audio_chunk_context import AudioChunkContext
from infrastructure.speech.recognizer.base import BaseSpeechRecognizer



class SpeechWorker:
    def __init__(
        self,
        camera_id: str,
        audio_bus,
        event_bus,
        recognizer: BaseSpeechRecognizer,
        window_seconds: float = 4.0,
        step_seconds: float = 1.0,
        min_audio_seconds: float = 1.5,
        duplicate_text_cooldown_seconds: float = 5.0,
        audio_overlay_state=None,

    ):
        self.camera_id = camera_id
        self.audio_bus = audio_bus
        self.event_bus = event_bus
        self.recognizer = recognizer
        self.audio_overlay_state = audio_overlay_state

        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.min_audio_seconds = float(min_audio_seconds)
        self.duplicate_text_cooldown_seconds = float(
            duplicate_text_cooldown_seconds
        )

        self.running = False

        self.buffer: Deque[AudioChunkContext] = deque()
        self.last_transcription_at = 0.0

        self.last_text = ""
        self.last_text_at = 0.0

    async def run(self):
        self.running = True
        queue = self.audio_bus.subscribe(self.camera_id)

        print(f"[SpeechWorker] {self.camera_id}: subscribed audio topic")

        try:
            while self.running:
                audio_ctx = await queue.get()

                # Chunks keep arriving while the recogniser runs; take all of
                # them so the next window is the newest, gap-free audio.
                for chunk in [audio_ctx, *self._drain(queue)]:
                    if chunk is None:
                        continue

                    if self.audio_overlay_state is not None:
                        self.audio_overlay_state.update_audio_chunk(chunk)

                    self.buffer.append(chunk)

                if not self.buffer:
                    continue

                self._trim_buffer()

                now = time.time()

                if now - self.last_transcription_at < self.step_seconds:
                    continue

                self.last_transcription_at = now

                pcm, ts_start, ts_end, sample_rate, session_id = self._build_window()

                if pcm is None:
                    continue

                duration = len(pcm) / float(sample_rate)

                if duration < self.min_audio_seconds:
                    continue

                audio_float = self._pcm_int16_to_float32(pcm)

                try:
                    result = await asyncio.to_thread(
                        self.recognizer.transcribe,
                        audio_float,
                        sample_rate,
                    )
                except Exception:
                    print(f"[SpeechWorker] {self.camera_id}: ASR failed")
                    traceback.print_exc()
                    continue

                text = str(result.get("text") or "").strip()

                if not text:
                    continue

                if self._is_duplicate_text(text, now):
                    continue

                self.last_text = text
                self.last_text_at = now

                event = {
                    "type": "speech.transcribed",
                    "camera_id": self.camera_id,
                    "text": text,
                    "timestamp": ts_end,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "sample_rate": sample_rate,
                    "language": result.get("language"),
                    "confidence": result.get("confidence"),
                    "stream_session_id": session_id,
                }

                if self.audio_overlay_state is not None:
                    self.audio_overlay_state.update_transcript(event)

                await self.event_bus.publish("speech.transcribed", event)

        except asyncio.CancelledError:
            pass

        except Exception:
            print(f"[SpeechWorker] {self.camera_id}: worker crashed")
            traceback.print_exc()

        finally:
            self.audio_bus.unsubscribe(self.camera_id, queue)
            print(f"[SpeechWorker] {self.camera_id}: stopped")

    def stop(self):
        self.running = False

    @staticmethod
    def _drain(queue: asyncio.Queue) -> list:
        chunks = []

        while True:
            try:
                chunks.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return chunks

    def _trim_buffer(self):
        if not self.buffer:
            return

        newest_end = self.buffer[-1].timestamp_end
        min_start = newest_end - self.window_seconds

        while self.buffer and self.buffer[0].timestamp_end < min_start:
            self.buffer.popleft()

    def _build_window(
        self,
    ) -> tuple[np.ndarray | None, float | None, float | None, int | None, str | None]:
        if not self.buffer:
            return None, None, None, None, None

        chunks = list(self.buffer)

        sample_rate = chunks[-1].sample_rate
        stream_session_id = chunks[-1].stream_session_id

        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.sample_rate == sample_rate
        ]

        if not valid_chunks:
            return None, None, None, None, None

        pcm = np.concatenate([chunk.pcm for chunk in valid_chunks])

        return (
            pcm,
            valid_chunks[0].timestamp_start,
            valid_chunks[-1].timestamp_end,
            sample_rate,
            stream_session_id,
        )

    def _pcm_int16_to_float32(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        return pcm.astype(np.float32) / 32768.0

    def _is_duplicate_text(self, text: str, now: float) -> bool:
        normalized = self._normalize_text(text)
        last_normalized = self._normalize_text(self.last_text)

        if not normalized:
            return True

        if normalized != last_normalized:
            return False

        return now - self.last_text_at < self.duplicate_text_cooldown_seconds

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().strip().split())