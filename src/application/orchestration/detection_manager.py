import asyncio

from domain.models.face_detection_options import FaceDetectorOptions
from infrastructure.ai_detection.providers.detector_provider import DetectorProvider
from application.orchestration.detection_worker import DetectionWorker


class DetectionManager:
    def __init__(self, frame_bus, event_bus, model_provider, settings, face_identity_store):
        self.frame_bus = frame_bus
        self.event_bus = event_bus
        self.settings = settings
        self.face_identity_store=face_identity_store
        self.model_provider = model_provider

        self.detector_provider = DetectorProvider(
            model_provider=self.model_provider,
            face_identity_store=self.face_identity_store,
            face_detector_options=FaceDetectorOptions(
                require_person=True,
                fallback_to_full_frame=False,
                use_upper_body_roi=True,
                model="hog",
                number_of_times_to_upsample=0,
                crop_upscale=1.2,
                full_frame_scale=0.75,
                min_face_size=18,
                num_jitters=1,
                debug=False,
            ),
        )
                
                
        self.tasks: list[asyncio.Task] = []
        self.workers: list[DetectionWorker] = []

    async def start(self):
        for camera_id, config in self.settings.registry.items():
            if not config.camera.active or not config.detections:
                continue

            worker = DetectionWorker(
                camera_id=camera_id,
                frame_bus=self.frame_bus,
                event_bus=self.event_bus,
                detector_provider=self.detector_provider,
                specs=config.detections,
                scene_tracking_enabled=getattr(config.camera, "scene_tracking", False),
                motion_roi_polygon=getattr(config.camera, "motion_mask_polygon", None),
            )

            self.workers.append(worker)

            task = asyncio.create_task(
                worker.run(),
                name=f"detection-worker-{camera_id}",
            )

            self.tasks.append(task)

            print(
                f"[DetectionManager] Worker started: {camera_id} | "
                f"scene_tracking={getattr(config.camera, 'scene_tracking', False)}"
            )

    async def stop(self):
        print("[DetectionManager] Stopping detection workers...")

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

        print("[DetectionManager] Detection workers stopped.")