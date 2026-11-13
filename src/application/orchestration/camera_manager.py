# application/managers/camera_manager.py

from __future__ import annotations

import asyncio

from application.services.camera_provider import CameraProvider
from infrastructure.cameras.camera_driver import CameraDriver


class CameraManager:
    def __init__(self, frame_bus, audio_bus, event_bus, settings, camera_stats=None):
        self.frame_bus = frame_bus
        self.audio_bus = audio_bus
        self.event_bus = event_bus
        self.settings = settings
        self.camera_stats = camera_stats

        self.drivers: list[CameraDriver] = []
        self.tasks: list[asyncio.Task] = []

    async def start_all(self):
        active_cameras = CameraProvider.get_active_cameras(self.settings)

        if not active_cameras:
            print("[CameraManager] No active cameras configured.")
            return

        print(f"[CameraManager] Starting {len(active_cameras)} camera(s)...")

        for camera_config in active_cameras:
            driver = CameraDriver(
                config=camera_config,
                frame_bus=self.frame_bus,
                audio_bus=self.audio_bus,
                audio_sample_rate=16000,
                audio_chunk_seconds=1.0,
                camera_stats=self.camera_stats,
            )

            self.drivers.append(driver)

            task = asyncio.create_task(
                driver.run(),
                name=f"camera-{camera_config.camera.id}",
            )

            task.add_done_callback(self._handle_task_completion)
            self.tasks.append(task)

            print(
                f"[CameraManager] Camera task started: "
                f"{camera_config.camera.id}"
            )

    async def stop_all(self):
        print("[CameraManager] Stopping cameras...")

        for driver in self.drivers:
            driver.stop()

        for task in self.tasks:
            if not task.done():
                task.cancel()

        if self.tasks:
            await asyncio.gather(
                *self.tasks,
                return_exceptions=True,
            )

        self.tasks.clear()
        self.drivers.clear()

        print("[CameraManager] All cameras stopped.")

    def _handle_task_completion(self, task: asyncio.Task):
        try:
            task.result()

        except asyncio.CancelledError:
            print(
                f"[CameraManager] Task cancelled: "
                f"{task.get_name()}"
            )

        except Exception as ex:
            print(
                f"[CameraManager] Task crashed: "
                f"{task.get_name()} | {ex}"
            )