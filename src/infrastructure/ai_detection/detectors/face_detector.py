import cv2
import numpy as np
import face_recognition

from domain.models.detection_type import DetectionType
from domain.models.detection_result import DetectionResult
from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.face_identity_store import FaceIdentityStore


class FaceDetector(BaseDetector):
    def __init__(
        self,
        model_provider=None,
        identity_store: FaceIdentityStore | None = None,
        require_person: bool = False,
        fallback_to_full_frame: bool = True,
        use_upper_body_roi: bool = True,
        model: str = "hog",
        number_of_times_to_upsample: int = 1,
        crop_upscale: float = 1.4,
        full_frame_scale: float = 0.75,
        min_face_size: int = 18,
        num_jitters: int = 1,
        debug: bool = False,
    ):
        self.identity_store = identity_store

        self.require_person = bool(require_person)
        self.fallback_to_full_frame = bool(fallback_to_full_frame)
        self.use_upper_body_roi = bool(use_upper_body_roi)

        self.model = str(model)
        self.number_of_times_to_upsample = int(number_of_times_to_upsample)
        self.num_jitters = int(num_jitters)

        self.crop_upscale = max(1.0, float(crop_upscale))
        self.full_frame_scale = max(0.1, min(1.0, float(full_frame_scale)))

        self.min_face_size = int(min_face_size)
        self.debug = bool(debug)

    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (255, 0, 255)

    @classmethod
    def get_display_name(cls) -> str:
        return "Twarz"

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict,
    ) -> list[DetectionResult]:
        if frame_data is None or frame_data.size == 0:
            return []

        rgb_frame = self._to_rgb_frame(frame_data)
        frame_height, frame_width = rgb_frame.shape[:2]

        person_boxes = context.get("detected_persons")
        has_person_boxes = person_boxes is not None and len(person_boxes) > 0

        detections: list[DetectionResult] = []
        recognized_faces: list[dict] = []
        unknown_faces: list[dict] = []

        if has_person_boxes:
            for person_box in person_boxes:
                box = self._extract_box(person_box)

                if box is None:
                    continue

                x1, y1, x2, y2 = self._clip_box(
                    box=box,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                if self.use_upper_body_roi:
                    person_h = y2 - y1
                    y2 = y1 + int(person_h * 0.58)

                region_detections, region_known, region_unknown = self._detect_faces_in_region(
                    rgb_frame=rgb_frame,
                    region_x1=x1,
                    region_y1=y1,
                    region_x2=x2,
                    region_y2=y2,
                    scale=self.crop_upscale,
                )

                detections.extend(region_detections)
                recognized_faces.extend(region_known)
                unknown_faces.extend(region_unknown)

        else:
            if self.debug:
                print("[FaceDetector] context has no detected_persons")

            if self.require_person or not self.fallback_to_full_frame:
                context["detected_faces"] = []
                context["recognized_faces"] = []
                context["unknown_faces"] = []
                return []

            region_detections, region_known, region_unknown = self._detect_faces_in_region(
                rgb_frame=rgb_frame,
                region_x1=0,
                region_y1=0,
                region_x2=frame_width,
                region_y2=frame_height,
                scale=self.full_frame_scale,
            )

            detections.extend(region_detections)
            recognized_faces.extend(region_known)
            unknown_faces.extend(region_unknown)

        context["detected_faces"] = detections
        context["recognized_faces"] = recognized_faces
        context["unknown_faces"] = unknown_faces

        if self.debug:
            person_count = len(person_boxes) if has_person_boxes else 0
            known_count = (
                self.identity_store.known_faces_count
                if self.identity_store is not None
                else 0
            )

            print(
                f"[FaceDetector] persons={person_count} "
                f"faces={len(detections)} "
                f"recognized={len(recognized_faces)} "
                f"unknown={len(unknown_faces)} "
                f"known_index={known_count}"
            )

        return detections

    def _to_rgb_frame(self, frame_data: np.ndarray) -> np.ndarray:
        if frame_data.dtype != np.uint8:
            frame_data = frame_data.astype(np.uint8)

        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb_frame)

    def _extract_box(self, box) -> tuple[int, int, int, int] | None:
        if box is None:
            return None

        if hasattr(box, "xyxy"):
            try:
                values = np.asarray(box.xyxy[0]).reshape(-1)

                if values.size >= 4:
                    return (
                        int(values[0]),
                        int(values[1]),
                        int(values[2]),
                        int(values[3]),
                    )
            except Exception:
                return None

        if all(hasattr(box, attr) for attr in ("x1", "y1", "x2", "y2")):
            try:
                return (
                    int(getattr(box, "x1")),
                    int(getattr(box, "y1")),
                    int(getattr(box, "x2")),
                    int(getattr(box, "y2")),
                )
            except Exception:
                return None

        if isinstance(box, dict):
            try:
                if all(key in box for key in ("x1", "y1", "x2", "y2")):
                    return (
                        int(box["x1"]),
                        int(box["y1"]),
                        int(box["x2"]),
                        int(box["y2"]),
                    )

                if "xyxy" in box:
                    values = np.asarray(box["xyxy"]).reshape(-1)

                    if values.size >= 4:
                        return (
                            int(values[0]),
                            int(values[1]),
                            int(values[2]),
                            int(values[3]),
                        )
            except Exception:
                return None

        if isinstance(box, (list, tuple, np.ndarray)):
            try:
                values = np.asarray(box).reshape(-1)

                if values.size >= 4:
                    return (
                        int(values[0]),
                        int(values[1]),
                        int(values[2]),
                        int(values[3]),
                    )
            except Exception:
                return None

        return None

    def _clip_box(
        self,
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box

        x1 = max(0, min(frame_width - 1, int(x1)))
        y1 = max(0, min(frame_height - 1, int(y1)))
        x2 = max(0, min(frame_width - 1, int(x2)))
        y2 = max(0, min(frame_height - 1, int(y2)))

        return x1, y1, x2, y2

    def _detect_faces_in_region(
        self,
        rgb_frame: np.ndarray,
        region_x1: int,
        region_y1: int,
        region_x2: int,
        region_y2: int,
        scale: float,
    ) -> tuple[list[DetectionResult], list[dict], list[dict]]:
        region_width = region_x2 - region_x1
        region_height = region_y2 - region_y1

        if region_width <= 0 or region_height <= 0:
            return [], [], []

        region = rgb_frame[region_y1:region_y2, region_x1:region_x2]

        if region.size == 0:
            return [], [], []

        region = np.ascontiguousarray(region)

        scale = float(scale)

        if scale != 1.0:
            resized_width = max(1, int(region_width * scale))
            resized_height = max(1, int(region_height * scale))

            detection_image = cv2.resize(
                region,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA,
            )
        else:
            detection_image = region

        detection_image = np.ascontiguousarray(detection_image)

        try:
            face_locations = face_recognition.face_locations(
                detection_image,
                number_of_times_to_upsample=self.number_of_times_to_upsample,
                model=self.model,
            )
        except Exception as exc:
            if self.debug:
                print(f"[FaceDetector] face_locations failed: {exc}")
            return [], [], []

        if not face_locations:
            return [], [], []

        try:
            face_encodings = face_recognition.face_encodings(
                detection_image,
                known_face_locations=face_locations,
                num_jitters=self.num_jitters,
                model="small",
            )
        except Exception as exc:
            if self.debug:
                print(f"[FaceDetector] face_encodings failed: {exc}")
            return [], [], []

        detections: list[DetectionResult] = []
        recognized_faces: list[dict] = []
        unknown_faces: list[dict] = []

        inv_scale = 1.0 / scale

        for face_location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location

            local_x1 = int(left * inv_scale)
            local_y1 = int(top * inv_scale)
            local_x2 = int(right * inv_scale)
            local_y2 = int(bottom * inv_scale)

            global_x1 = region_x1 + local_x1
            global_y1 = region_y1 + local_y1
            global_x2 = region_x1 + local_x2
            global_y2 = region_y1 + local_y2

            face_w = global_x2 - global_x1
            face_h = global_y2 - global_y1

            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            match = None

            if self.identity_store is not None:
                match = self.identity_store.match(face_encoding)

            if match is not None:
                label = match.display_name
                confidence = match.confidence

                recognized_faces.append(
                    {
                        "person_id": match.person_id,
                        "display_name": match.display_name,
                        "distance": match.distance,
                        "confidence": match.confidence,
                        "x1": global_x1,
                        "y1": global_y1,
                        "x2": global_x2,
                        "y2": global_y2,
                    }
                )
            else:
                if self.identity_store is None:
                    label = "Twarz"
                else:
                    label = "Nieznana twarz"

                confidence = 1.0

                unknown_faces.append(
                    {
                        "x1": global_x1,
                        "y1": global_y1,
                        "x2": global_x2,
                        "y2": global_y2,
                    }
                )

            detections.append(
                DetectionResult(
                    detection_type=DetectionType.FACE.value,
                    x1=global_x1,
                    y1=global_y1,
                    x2=global_x2,
                    y2=global_y2,
                    confidence=confidence,
                    label=label,
                )
            )

        return detections, recognized_faces, unknown_faces