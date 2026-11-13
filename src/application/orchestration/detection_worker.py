import asyncio
import math
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from domain.models.archived_detection_state import ArchivedDetectionState
from domain.models.frame_context import FrameContext
from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.detectors.motion_detector import MotionFilter
from infrastructure.ai_detection.tracking.tracker_decorator import TrackingDetectorDecorator
from infrastructure.archive.scene_tracker import SceneTracker

if TYPE_CHECKING:
    from infrastructure.ai_detection.providers.detector_provider import DetectorProvider


# Detector and model instances are shared between cameras, and Ultralytics
# models are not safe to call from several threads at once. On a CPU, running
# inferences one at a time is also faster than letting them fight for cores.
_INFERENCE_LOCK = threading.Lock()


def _run_detector(detector, frame, context) -> tuple[list, float, float]:
    """Runs one detector under the shared lock; returns (detections, wait_ms, infer_ms)."""
    t0 = time.perf_counter()

    with _INFERENCE_LOCK:
        t1 = time.perf_counter()
        detections = detector.execute(frame, context)
        t2 = time.perf_counter()

    return detections, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0




class DetectionWorker:
    def __init__(
        self,
        camera_id: str,
        frame_bus,
        event_bus,
        detector_provider: "DetectorProvider",
        specs,
        scene_tracking_enabled: bool = False,
        motion_roi_polygon: list | None = None,
    ):
        self.camera_id = camera_id
        self.frame_bus = frame_bus
        self.event_bus = event_bus
        self.detector_provider = detector_provider
        self.detectors: dict[str, BaseDetector] = {}
        self.specs = list(specs)

        for spec in self.specs:
            d_type = self._spec_type(spec)
            base = self.detector_provider.get_detector(d_type) # czysty AI

            if getattr(spec, "tracking", False):
                base = TrackingDetectorDecorator(base)

            self.detectors[d_type] = base

            

        self.running = False
        self.motion_filter = MotionFilter()
        # Motion only counts inside this polygon (frame pixel coordinates);
        # empty means the whole frame.
        self.motion_roi_polygon = list(motion_roi_polygon or [])

        # For static cameras keep a single logical scene and do not run visual
        # scene detection. For PTZ cameras set scene_tracking_enabled=True.
        self.scene_tracking_enabled = scene_tracking_enabled
        self.scene_tracker = SceneTracker(debug=False) if scene_tracking_enabled else None
        self.current_scene_id = (
            f"{camera_id}_scene_0000"
            if scene_tracking_enabled
            else f"{camera_id}_static_scene"
        )

        # Render cache: detectors run at ai_fps, camera frames arrive faster.
        # This keeps boxes visible between AI executions.
        self.last_detections_by_type: dict[str, list] = {}
        self.last_detection_time_by_type: dict[str, float] = {}
        self.detection_ttl_sec = 1.25

        # Archive/event dedupe: do not publish/save the same stationary box over
        # and over. A new event is emitted only when the scene, count, label set,
        # or box geometry changes enough.
        self.last_archived_state_by_type: dict[str, ArchivedDetectionState] = {}
        self.absent_since_by_type: dict[str, float] = {}
        self.absence_reset_sec = 5.0

        # Geometry thresholds for deciding whether boxes are effectively the same.
        self.same_box_iou_threshold = 0.70
        self.same_box_center_shift_ratio = 0.025
        self.same_box_area_change_ratio = 0.25

        # Disabled by default. Set to e.g. 600.0 if you want one low-frequency
        # confirmation event for unchanged detections every 10 minutes.
        self.same_state_heartbeat_sec = 0.0

    async def run(self):
        self.running = True
        queue = self.frame_bus.subscribe(self.camera_id)

        last_execution_times = {
            self._spec_type(spec): 0.0
            for spec in self.specs
        }

        print(
            f"[DetectionWorker] {self.camera_id}: subscribed raw topic "
            f"'{self.camera_id}', publishing AI topic '{self.camera_id}_ai', "
            f"scene_tracking_enabled={self.scene_tracking_enabled}"
        )

        try:
            while self.running:
                frame_ctx = await queue.get()

                if frame_ctx is None or getattr(frame_ctx, "is_empty", False):
                    continue

                now = time.time()
                frame_timestamp = getattr(frame_ctx, "timestamp", now)

                scene_id, scene_changed = self._refresh_scene(frame_ctx, now)

                if scene_changed:
                    self._clear_render_cache()
                    self.motion_filter.reset(self.camera_id)

                has_motion = self.motion_filter.has_significant_motion(
                    self.camera_id,
                    frame_ctx.frame_data,
                    roi_polygon=self.motion_roi_polygon,
                )

                context: dict[str, Any] = {
                    "clean_frame": frame_ctx.frame_data
                }

                frame_height = int(frame_ctx.frame_data.shape[0])
                frame_width = int(frame_ctx.frame_data.shape[1])

                # No motion: detectors are skipped. Absence is only concluded from a
                # detector run that finds nothing, so a parked car keeps its
                # archive state while the scene is still.
                if has_motion:

                    for spec in self.specs:
                        detection_type = self._spec_type(spec)
                        ai_fps = self._get_ai_fps(spec)
                        interval = 1.0 / ai_fps

                        if now - last_execution_times[detection_type] < interval:
                            continue

                        last_execution_times[detection_type] = now
                        detector = self.detectors[detection_type]

                        try:
                            detections, wait_ms, infer_ms = await asyncio.to_thread(
                                _run_detector,
                                detector,
                                frame_ctx.frame_data,
                                context,
                            )

                            if infer_ms + wait_ms > 150:
                                print(
                                    f"[DetectionWorker] {self.camera_id}: "
                                    f"{detection_type} took {infer_ms:.1f} ms "
                                    f"(+{wait_ms:.0f} ms queued), "
                                    f"detections={len(detections or [])}"
                                )
                        except Exception:
                            print(f"[DetectionWorker] {self.camera_id}: detector failed: {detection_type}")
                            traceback.print_exc()
                            continue

                        detections = detections or []
                        self._update_render_cache(detection_type, detections, now)

                        if not detections:
                            self._handle_absence(detection_type, now)
                            continue

                        self.absent_since_by_type.pop(detection_type, None)
                        boxes = [self._serialize_detection(detection) for detection in detections]

                        should_publish, reason = self._should_publish_detection_event(
                            detection_type=detection_type, scene_id=scene_id, boxes=boxes,
                            frame_width=frame_width, frame_height=frame_height, now=now,
                        )

                        if not should_publish:
                            continue

                        event = self._build_detection_event(
                            frame_ctx=frame_ctx, detection_type=detection_type, boxes=boxes,
                            scene_id=scene_id, timestamp=frame_timestamp, frame_width=frame_width,
                            frame_height=frame_height, reason=reason,
                        )
                        event["snapshot_frame"] = frame_ctx.frame_data.copy()
                        event["snapshot_reason"] = reason

                        await self.event_bus.publish("detections", event)

                visible_detections = self._get_visible_detections(now)

                ai_frame_ctx = FrameContext(
                    camera_id=frame_ctx.camera_id,
                    frame_id=frame_ctx.frame_id,
                    frame_data=frame_ctx.frame_data,
                    detections=visible_detections,
                    timestamp=frame_timestamp,
                    stream_session_id=getattr(frame_ctx, "stream_session_id", None),
                    scene_id=scene_id,
                )

                await self.frame_bus.publish(
                    f"{self.camera_id}_ai",
                    ai_frame_ctx,
                )


        except asyncio.CancelledError:
            pass

        except Exception:
            print(f"[DetectionWorker] {self.camera_id}: worker crashed")
            traceback.print_exc()

        finally:
            self.frame_bus.unsubscribe(self.camera_id, queue)
            print(f"[DetectionWorker] {self.camera_id}: stopped")

    def stop(self):
        self.running = False

    def _refresh_scene(
        self,
        frame_ctx: FrameContext,
        now: float,
    ) -> tuple[str, bool]:
        if self.scene_tracker is None:
            frame_ctx.scene_id = self.current_scene_id
            return self.current_scene_id, False

        try:
            scene_id, scene_changed = self.scene_tracker.update(
                self.camera_id,
                frame_ctx.frame_data,
                now=now,
            )
        except Exception:
            print(f"[DetectionWorker] {self.camera_id}: scene tracker failed")
            traceback.print_exc()
            scene_id = self.current_scene_id
            scene_changed = False

        self.current_scene_id = scene_id
        frame_ctx.scene_id = scene_id

        return scene_id, scene_changed

    def _update_render_cache(
        self,
        detection_type: str,
        detections: list,
        now: float,
    ) -> None:
        self.last_detections_by_type[detection_type] = detections
        self.last_detection_time_by_type[detection_type] = now

    def _get_visible_detections(self, now: float) -> list:
        visible = []

        for detection_type, detections in self.last_detections_by_type.items():
            detected_at = self.last_detection_time_by_type.get(detection_type, 0.0)

            if now - detected_at <= self.detection_ttl_sec:
                visible.extend(detections)

        return visible

    def _clear_render_cache(self) -> None:
        self.last_detections_by_type.clear()
        self.last_detection_time_by_type.clear()

    def _handle_absence(
        self,
        detection_type: str,
        now: float,
    ) -> None:
        if detection_type not in self.last_archived_state_by_type:
            return

        absent_since = self.absent_since_by_type.setdefault(detection_type, now)

        if now - absent_since >= self.absence_reset_sec:
            del self.last_archived_state_by_type[detection_type]
            self.absent_since_by_type.pop(detection_type, None)
            print(
                f"[DetectionWorker] {self.camera_id}: reset archive state "
                f"for {detection_type} after absence"
            )

    def _should_publish_detection_event(
        self,
        detection_type: str,
        scene_id: str,
        boxes: list[dict],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> tuple[bool, str]:
        previous = self.last_archived_state_by_type.get(detection_type)

        if previous is None:
            self._store_archived_state(detection_type, scene_id, boxes, now)
            return True, "new_detection"

        previous.last_seen_ts = now

        if previous.scene_id != scene_id:
            self._store_archived_state(detection_type, scene_id, boxes, now)
            return True, "scene_changed"

        if len(previous.boxes) != len(boxes):
            self._store_archived_state(detection_type, scene_id, boxes, now)
            return True, "count_changed"

        if self._boxes_changed(
            previous_boxes=previous.boxes,
            current_boxes=boxes,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            self._store_archived_state(detection_type, scene_id, boxes, now)
            return True, "boxes_changed"

        if (
            self.same_state_heartbeat_sec > 0
            and now - previous.last_event_ts >= self.same_state_heartbeat_sec
        ):
            self._store_archived_state(detection_type, scene_id, boxes, now)
            return True, "same_state_heartbeat"

        return False, "same_boxes"

    def _store_archived_state(
        self,
        detection_type: str,
        scene_id: str,
        boxes: list[dict],
        now: float,
    ) -> None:
        self.last_archived_state_by_type[detection_type] = ArchivedDetectionState(
            scene_id=scene_id,
            boxes=[dict(box) for box in boxes],
            last_event_ts=now,
            last_seen_ts=now,
        )

    def _boxes_changed(
        self,
        previous_boxes: list[dict],
        current_boxes: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        unused_previous = set(range(len(previous_boxes)))

        for current in current_boxes:
            best_index = None
            best_score = -1.0

            for index in unused_previous:
                previous = previous_boxes[index]

                if not self._labels_compatible(previous, current):
                    continue

                iou = self._iou(previous, current)
                center_ratio = self._center_shift_ratio(
                    previous,
                    current,
                    frame_width,
                    frame_height,
                )
                area_ratio = self._area_change_ratio(previous, current)

                similar = (
                    iou >= self.same_box_iou_threshold
                    or (
                        center_ratio <= self.same_box_center_shift_ratio
                        and area_ratio <= self.same_box_area_change_ratio
                    )
                )

                if not similar:
                    continue

                score = iou - center_ratio

                if score > best_score:
                    best_score = score
                    best_index = index

            if best_index is None:
                return True

            unused_previous.remove(best_index)

        return bool(unused_previous)

    def _labels_compatible(
        self,
        previous: dict,
        current: dict,
    ) -> bool:
        previous_label = previous.get("label") or previous.get("type") or ""
        current_label = current.get("label") or current.get("type") or ""

        return str(previous_label) == str(current_label)

    def _iou(
        self,
        a: dict,
        b: dict,
    ) -> float:
        ax1, ay1, ax2, ay2 = self._box_coords(a)
        bx1, by1, bx2, by2 = self._box_coords(b)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0.0, ix2 - ix1)
        intersection_height = max(0.0, iy2 - iy1)
        intersection = intersection_width * intersection_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _center_shift_ratio(
        self,
        a: dict,
        b: dict,
        frame_width: int,
        frame_height: int,
    ) -> float:
        ax1, ay1, ax2, ay2 = self._box_coords(a)
        bx1, by1, bx2, by2 = self._box_coords(b)

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        distance = math.hypot(acx - bcx, acy - bcy)
        diagonal = math.hypot(float(frame_width), float(frame_height))

        if diagonal <= 0:
            return 1.0

        return distance / diagonal

    def _area_change_ratio(
        self,
        a: dict,
        b: dict,
    ) -> float:
        ax1, ay1, ax2, ay2 = self._box_coords(a)
        bx1, by1, bx2, by2 = self._box_coords(b)

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        if area_a <= 0 or area_b <= 0:
            return 1.0

        return abs(area_a - area_b) / max(area_a, area_b)

    def _box_coords(self, box: dict) -> tuple[float, float, float, float]:
        return (
            float(box.get("x1", 0.0)),
            float(box.get("y1", 0.0)),
            float(box.get("x2", 0.0)),
            float(box.get("y2", 0.0)),
        )

    def _build_detection_event(
        self,
        frame_ctx: FrameContext,
        detection_type: str,
        boxes: list[dict],
        scene_id: str,
        timestamp: float,
        frame_width: int,
        frame_height: int,
        reason: str,
    ) -> dict:
        return {
            "camera_id": self.camera_id,
            "stream_session_id": getattr(frame_ctx, "stream_session_id", None),
            "frame_id": frame_ctx.frame_id,
            "frame_uid": getattr(frame_ctx, "frame_uid", None),
            "scene_id": scene_id,
            "detection_type": detection_type,
            "count": len(boxes),
            "timestamp": timestamp,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "bounding_boxes": boxes,
            "event_reason": reason,
        }

    def _serialize_detection(self, detection) -> dict:
        x1 = int(detection.x1)
        y1 = int(detection.y1)
        x2 = int(detection.x2)
        y2 = int(detection.y2)

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": int((x1 + x2) / 2),
            "cy": int((y1 + y2) / 2),
            "confidence": float(detection.confidence),
            "label": detection.label,
            "type": detection.detection_type,
        }

    def _get_ai_fps(self, spec) -> float:
        value = getattr(spec, "ai_fps", 3)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 3.0

        return max(0.1, value)

    def _spec_type(self, spec) -> str:
        detection_type = getattr(spec, "detection_type", spec)

        if hasattr(detection_type, "value"):
            return str(detection_type.value)

        return str(detection_type)