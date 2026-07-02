import asyncio

from application.messaging.event_bus import EventBus
from application.messaging.frame_bus import FrameBus
from application.messaging.audio_bus import AudioBus

from application.orchestration.camera_manager import CameraManager
from application.orchestration.detection_manager import DetectionManager
from application.orchestration.speech_manager import SpeechManager

from application.runtime.audio_overlay_state import AudioOverlayState
from infrastructure.ai_detection.face_identity_store import FaceIdentityStore
from infrastructure.ai_detection.providers.model_provider import DetectionModelProvider
from infrastructure.archive.archive_writer import ArchiveWriter

from infrastructure.automation.automation_orchestrator import AutomationRuleEngine
from infrastructure.automation.controllers.gate_controler import GateControllerWorker
from infrastructure.automation.multimodal_gate_automation_engine import (
    MultimodalGateAutomationEngine,
)
from infrastructure.speech.recognizer.faster_whisper_recognizer import (
    FasterWhisperSpeechRecognizer,
)


class AppOrchestrator:

    def __init__(self, settings):
        self.settings = settings

        self.frame_bus = FrameBus()
        self.audio_bus = AudioBus()
        self.event_bus = EventBus()

        self.audio_overlay_state = AudioOverlayState()

        self.face_identity_store = FaceIdentityStore(
            known_faces_dir="data/faces/known",
            index_path="data/faces/known_faces.pkl",
            display_names_path="data/faces/display_names.json",
            tolerance=0.48,
            auto_reload=True,
            debug=False,
        )

        self.model_provider = DetectionModelProvider()

        self.camera_manager = CameraManager(
            frame_bus=self.frame_bus,
            audio_bus=self.audio_bus,
            event_bus=self.event_bus,
            settings=settings,
        )

        self.detection_manager = DetectionManager(
            frame_bus=self.frame_bus,
            event_bus=self.event_bus,
            model_provider=self.model_provider,
            settings=settings,
            face_identity_store=self.face_identity_store,
        )

        self.speech_manager = SpeechManager(
            audio_bus=self.audio_bus,
            event_bus=self.event_bus,
            settings=settings,
            audio_overlay_state=self.audio_overlay_state,
            recognizer_factory=lambda camera_config: FasterWhisperSpeechRecognizer(
                model_size="small",
                device="cpu",
                compute_type="int8",
                language="pl",
            ),
        )

        self.archive_writer = ArchiveWriter(event_bus=self.event_bus)

        self.gate_controller_worker = GateControllerWorker(
            event_bus=self.event_bus,
            gate_controller=None,
            command_topic="gate.command.open",
        )

        self.running = False

        self._rule_engine_tasks = []
        self._multimodal_engine_tasks = []
        self._background_tasks = []

    async def start(self):
        if self.running:
            return

        self.running = True

        print("[Orchestrator] Starting systems...")

        await self._start_face_identity_store()

        self._start_rule_engines()
        self._start_multimodal_gate_engines()

        self._background_tasks.append(
            asyncio.create_task(
                self.gate_controller_worker.run(),
                name="gate-controller-worker",
            )
        )

        self._background_tasks.append(
            asyncio.create_task(
                self._listen_to_events(),
                name="event-console-listener",
            )
        )

        self._background_tasks.append(
            asyncio.create_task(
                self._listen_to_speech_events(),
                name="speech-console-listener",
            )
        )

        self._background_tasks.append(
            asyncio.create_task(
                self.archive_writer.run(),
                name="archive-writer",
            )
        )

        await self.detection_manager.start()
        await self.speech_manager.start_all()
        await self.camera_manager.start_all()

        print("[Orchestrator] Startup complete.")

    async def stop(self):
        if not self.running:
            return

        print("[Orchestrator] Stopping systems...")

        self.running = False

        self.gate_controller_worker.stop()

        for task in self._rule_engine_tasks:
            task.cancel()

        for task in self._multimodal_engine_tasks:
            task.cancel()

        for task in self._background_tasks:
            task.cancel()

        all_tasks = (
            self._rule_engine_tasks
            + self._multimodal_engine_tasks
            + self._background_tasks
        )

        if all_tasks:
            await asyncio.gather(
                *all_tasks,
                return_exceptions=True,
            )

        await self.camera_manager.stop_all()
        await self.speech_manager.stop_all()
        await self.detection_manager.stop()

        self._rule_engine_tasks.clear()
        self._multimodal_engine_tasks.clear()
        self._background_tasks.clear()

        print("[Orchestrator] Shutdown complete.")

    async def _start_face_identity_store(self):
        if not self._has_face_detection_configured():
            print("[FaceIdentityStore] skipped: face_detection is not configured")
            return

        print("[FaceIdentityStore] starting...")

        await asyncio.to_thread(
            self.face_identity_store.start,
            rebuild_if_missing=True,
            rebuild_if_stale=True,
        )

    def _has_face_detection_configured(self) -> bool:
        detection_configs = getattr(self.settings, "detection_configs", {}) or {}

        for camera_id, specs in detection_configs.items():
            config = self.settings.registry.get(camera_id)

            if config is not None and not getattr(config.camera, "active", False):
                continue

            for spec in specs or []:
                detection_type = getattr(spec, "detection_type", spec)
                detection_type = getattr(detection_type, "value", detection_type)

                if str(detection_type) == "face_detection":
                    return True

        return False

    def _start_rule_engines(self):
        for cam_id, config in self.settings.registry.items():
            if not config.camera.active or not config.camera.rules:
                continue

            has_tripwire = any(
                rule.get("type") == "tripwire"
                for rule in config.camera.rules
            )

            if not has_tripwire:
                continue

            engine = AutomationRuleEngine(
                frame_bus=self.frame_bus,
                event_bus=self.event_bus,
                camera_config=config,
            )

            worker_topic = f"{cam_id}_ai"

            task = asyncio.create_task(
                engine.start_processing(worker_topic),
                name=f"rule-engine-{cam_id}",
            )

            self._rule_engine_tasks.append(task)

    def _start_multimodal_gate_engines(self):
        for cam_id, config in self.settings.registry.items():
            if not config.camera.active or not config.camera.rules:
                continue

            has_multimodal_gate_rule = any(
                rule.get("type") == "multimodal_gate_open"
                for rule in config.camera.rules
            )

            if not has_multimodal_gate_rule:
                continue

            engine = MultimodalGateAutomationEngine(
                event_bus=self.event_bus,
                camera_config=config,
            )

            task = asyncio.create_task(
                engine.start_processing(),
                name=f"multimodal-gate-engine-{cam_id}",
            )

            self._multimodal_engine_tasks.append(task)

    async def _listen_to_events(self):
        queue = self.event_bus.subscribe("detections")

        try:
            while self.running:
                event = await queue.get()
                camera_id = event.get("camera_id")
                det_type = event.get("detection_type")
                count = event.get("count")

                if count > 0:
                    print(
                        f"🚨 [AI ALERT] Kamera: {camera_id} | "
                        f"Wykryto: {det_type} | Ilość: {count}"
                    )
                else:
                    print(
                        f"🔍 [AI Spokój] Kamera: {camera_id} | "
                        f"Brak obiektów typu {det_type}"
                    )

        except asyncio.CancelledError:
            pass

        finally:
            self.event_bus.unsubscribe("detections", queue)

    async def _listen_to_speech_events(self):
        queue = self.event_bus.subscribe("speech.transcribed")

        try:
            while self.running:
                event = await queue.get()
                camera_id = event.get("camera_id")
                text = event.get("text")
                confidence = event.get("confidence")

                self.audio_overlay_state.update_transcript(event)

                print(
                    f"🎙️ [SPEECH] Kamera: {camera_id} | "
                    f"Tekst: {text!r} | Confidence: {confidence}"
                )

        except asyncio.CancelledError:
            pass

        finally:
            self.event_bus.unsubscribe("speech.transcribed", queue)