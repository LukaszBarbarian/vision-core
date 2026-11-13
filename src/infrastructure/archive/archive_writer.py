import asyncio
import csv
from datetime import datetime
from pathlib import Path

import cv2

from infrastructure.rendering.overlay import box_thickness, draw_info_bar, draw_label


class ArchiveWriter:
    def __init__(
        self,
        event_bus,
        log_dir: str = "archive_logs",
        draw_boxes_on_snapshots: bool = True,
    ):
        self.event_bus = event_bus
        self.log_dir = Path(log_dir)
        self.snapshots_dir = self.log_dir / "snapshots"
        self.csv_path = self.log_dir / "detections_long.csv"
        self.draw_boxes_on_snapshots = draw_boxes_on_snapshots

        self.running = True

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    async def run(self):
        queue = self.event_bus.subscribe("detections")
        print(
            "[ArchiveWriter] started | "
            f"csv={self.csv_path.resolve()} | "
            f"snapshots={self.snapshots_dir.resolve()}"
        )

        try:
            while self.running:
                event = await queue.get()
                await asyncio.to_thread(self._write_event, event)

        except asyncio.CancelledError:
            pass

        finally:
            self.event_bus.unsubscribe("detections", queue)
            print("[ArchiveWriter] stopped")

    def stop(self):
        self.running = False

    def _write_event(self, event: dict):
        camera_id = event.get("camera_id") or "unknown_camera"
        detection_type = event.get("detection_type") or "unknown_detection"
        timestamp = float(event.get("timestamp") or 0.0)
        frame_id = event.get("frame_id")
        stream_session_id = event.get("stream_session_id") or "no-session"
        scene_id = event.get("scene_id") or "unknown_scene"
        frame_width = event.get("frame_width")
        frame_height = event.get("frame_height")
        boxes = event.get("bounding_boxes", []) or []

        frame_uid = event.get("frame_uid") or f"{camera_id}:{stream_session_id}:{frame_id}"
        readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        snapshot_path = self._save_snapshot_if_available(
            event=event,
            camera_id=camera_id,
            detection_type=detection_type,
            stream_session_id=stream_session_id,
            frame_uid=frame_uid,
            boxes=boxes,
        )

        self._append_rows(
            readable_time=readable_time,
            camera_id=camera_id,
            stream_session_id=stream_session_id,
            frame_id=frame_id,
            frame_uid=frame_uid,
            scene_id=scene_id,
            detection_type=detection_type,
            frame_width=frame_width,
            frame_height=frame_height,
            snapshot_path=snapshot_path,
            boxes=boxes,
        )

    def _save_snapshot_if_available(
        self,
        event: dict,
        camera_id: str,
        detection_type: str,
        stream_session_id: str,
        frame_uid: str,
        boxes: list[dict],
    ) -> str:
        frame = event.get("snapshot_frame")

        if frame is None:
            return ""

        safe_frame_uid = self._safe_name(frame_uid)
        safe_detection_type = self._safe_name(detection_type)

        snapshot_dir = self.snapshots_dir / camera_id / stream_session_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = snapshot_dir / f"{safe_frame_uid}_{safe_detection_type}.jpg"

        image = frame.copy()

        if self.draw_boxes_on_snapshots:
            self._draw_boxes(image, boxes, detection_type)
            draw_info_bar(image, self._snapshot_caption(event, camera_id, detection_type, boxes))

        ok = cv2.imwrite(
            str(snapshot_file),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )

        if not ok:
            print(f"[ArchiveWriter] failed to save snapshot: {snapshot_file.resolve()}")
            return ""

        print(f"[ArchiveWriter] saved detection snapshot: {snapshot_file.resolve()}")
        return str(snapshot_file)

    def _append_rows(
        self,
        readable_time: str,
        camera_id: str,
        stream_session_id: str,
        frame_id,
        frame_uid: str,
        scene_id: str,
        detection_type: str,
        frame_width,
        frame_height,
        snapshot_path: str,
        boxes: list[dict],
    ) -> None:
        file_exists = self.csv_path.exists()

        fieldnames = [
            "timestamp",
            "camera_id",
            "stream_session_id",
            "frame_id",
            "frame_uid",
            "scene_id",
            "detection_type",
            "label",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "frame_width",
            "frame_height",
            "snapshot_path",
        ]

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for box in boxes:
                writer.writerow({
                    "timestamp": readable_time,
                    "camera_id": camera_id,
                    "stream_session_id": stream_session_id,
                    "frame_id": frame_id,
                    "frame_uid": frame_uid,
                    "scene_id": scene_id,
                    "detection_type": detection_type,
                    "label": box.get("label"),
                    "confidence": box.get("confidence"),
                    "x1": box.get("x1"),
                    "y1": box.get("y1"),
                    "x2": box.get("x2"),
                    "y2": box.get("y2"),
                    "cx": box.get("cx"),
                    "cy": box.get("cy"),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "snapshot_path": snapshot_path,
                })

    def _draw_boxes(
        self,
        image,
        boxes: list[dict],
        detection_type: str,
    ) -> None:
        color = self._color_for_detection_type(detection_type)

        height, width = image.shape[:2]

        for box in boxes:
            try:
                x1 = max(0, min(width - 1, int(box.get("x1", 0))))
                y1 = max(0, min(height - 1, int(box.get("y1", 0))))
                x2 = max(0, min(width - 1, int(box.get("x2", 0))))
                y2 = max(0, min(height - 1, int(box.get("y2", 0))))
            except (TypeError, ValueError):
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            label = box.get("label") or detection_type
            confidence = box.get("confidence")

            if confidence is not None:
                try:
                    label = f"{label}: {float(confidence):.2f}"
                except (TypeError, ValueError):
                    pass

            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness(image))
            draw_label(image, label, x1, y1, color)

    def _snapshot_caption(self, event: dict, camera_id: str, detection_type: str, boxes: list[dict]) -> str:
        """Camera, detector, object count, why the event was archived and when."""
        count = len(boxes)
        parts = [camera_id, detection_type, f"{count} object" + ("" if count == 1 else "s")]

        reason = event.get("event_reason")
        if reason:
            parts.append(str(reason))

        timestamp = event.get("timestamp")
        if timestamp:
            parts.append(datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S"))

        return " | ".join(parts)

    def _color_for_detection_type(self, detection_type: str) -> tuple[int, int, int]:
        colors = {
            "chicken_detection": (0, 165, 255),
            "person_detection": (255, 0, 0),
            "face_detection": (255, 0, 255),
            "car_detection": (0, 255, 255),
            "anpr_detection": (255, 255, 0),
        }
        return colors.get(detection_type, (0, 255, 0))

    def _safe_name(self, value) -> str:
        return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")