# application/managers/speech_manager.py

from __future__ import annotations

import asyncio
import traceback
from typing import Callable

from application.services.camera_provider import CameraProvider
from infrastructure.speech.recognizer.base import BaseSpeechRecognizer
from infrastructure.speech.speach_worker import SpeechWorker


class SpeechManager:
    def __init__(
        self,
        audio_bus,
        event_bus,
        settings,
        recognizer_factory: Callable[[object], BaseSpeechRecognizer],
        audio_overlay_state=None,

    ):
        self.audio_bus = audio_bus
        self.event_bus = event_bus
        self.settings = settings
        self.recognizer_factory = recognizer_factory
        self.audio_overlay_state = audio_overlay_state

        self.workers: list[SpeechWorker] = []
        self.tasks: list[asyncio.Task] = []

    async def start_all(self):
        active_cameras = CameraProvider.get_active_cameras(self.settings)

        speech_cameras = [
            camera_config
            for camera_config in active_cameras
            if bool(getattr(camera_config.camera, "audio_enabled", False))
        ]

        if not speech_cameras:
            print("[SpeechManager] No audio-enabled cameras configured.")
            return

        print(f"[SpeechManager] Starting {len(speech_cameras)} speech worker(s)...")

        for camera_config in speech_cameras:
            camera_id = camera_config.camera.id

            try:
                recognizer = self.recognizer_factory(camera_config)
            except Exception:
                print(f"[SpeechManager] Failed to create recognizer for {camera_id}")
                traceback.print_exc()
                continue

            worker = SpeechWorker(
                camera_id=camera_id,
                audio_bus=self.audio_bus,
                event_bus=self.event_bus,
                recognizer=recognizer,
                window_seconds=3.0,
                step_seconds=1.0,
                min_audio_seconds=1.5,
                audio_overlay_state=self.audio_overlay_state,

            )

            self.workers.append(worker)

            task = asyncio.create_task(
                worker.run(),
                name=f"speech-{camera_id}",
            )

            task.add_done_callback(self._handle_task_completion)
            self.tasks.append(task)

            print(f"[SpeechManager] Speech task started: {camera_id}")

    async def stop_all(self):
        print("[SpeechManager] Stopping speech workers...")

        for worker in self.workers:
            worker.stop()

        for task in self.tasks:
            if not task.done():
                task.cancel()

        if self.tasks:
            await asyncio.gather(
                *self.tasks,
                return_exceptions=True,
            )

        self.tasks.clear()
        self.workers.clear()

        print("[SpeechManager] All speech workers stopped.")

    def _handle_task_completion(self, task: asyncio.Task):
        try:
            task.result()

        except asyncio.CancelledError:
            print(f"[SpeechManager] Task cancelled: {task.get_name()}")

        except Exception as ex:
            print(f"[SpeechManager] Task crashed: {task.get_name()} | {ex}")