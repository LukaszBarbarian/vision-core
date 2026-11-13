import cv2
import numpy as np

from domain.models.frame_context import FrameContext
from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.rendering.overlay import box_thickness, draw_info_bar, draw_label, ui_scale


class FrameRenderer:
    def __init__(self, detector_provider=None, audio_overlay_state=None):
        self.detector_provider = detector_provider
        self.audio_overlay_state = audio_overlay_state

    def render(
        self,
        frame_ctx: FrameContext,
        visible_types: set[str] | None = None,
        config=None,
        status_text: str | None = None,
    ) -> np.ndarray:
        image = frame_ctx.frame_data.copy()
        detections = getattr(frame_ctx, "detections", []) or []
        height, width = image.shape[:2]
        visible_types = visible_types or set()
        thickness = box_thickness(image)
        dot = max(4, round(4 * ui_scale(image)))

        # 1. RYSOWANIE POLIGONU MASKI RUCHU (Transparent Polygon)
        if config and hasattr(config.camera, 'motion_mask_polygon') and config.camera.motion_mask_polygon:
            image = self._draw_motion_mask(image, config.camera.motion_mask_polygon)

        # 2. RYSOWANIE LINII TRIPWIRE
        if config and hasattr(config.camera, 'rules') and config.camera.rules:
            self._draw_tripwire_rules(image, config.camera.rules)

        # 3. RYSOWANIE DETEKCJI
        for detection in detections:
            detection_type = str(detection.detection_type)

            if visible_types and detection_type not in visible_types:
                continue

            x1 = max(0, min(width - 1, int(detection.x1)))
            y1 = max(0, min(height - 1, int(detection.y1)))
            x2 = max(0, min(width - 1, int(detection.x2)))
            y2 = max(0, min(height - 1, int(detection.y2)))

            if x2 <= x1 or y2 <= y1:
                continue
                
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bottom_y = int(y2)

            # Pobieramy bazowy kolor detektora
            color = self._get_color(detection_type)
            label = self._format_label(detection)

            # The rule engine marks a detection whose ground point crossed a tripwire.
            if getattr(detection, "triggered", False):
                color = (0, 0, 255)
                label = f"[CROSSED] {label}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(image, (cx, cy), dot, color, -1)
            cv2.circle(image, (cx, bottom_y), dot + 1, (0, 0, 255), -1)  # ground point

            draw_label(image, label, x1, y1, color)

        if status_text:
            draw_info_bar(image, status_text, corner="top-right")

        self._draw_audio_overlay(image=image, camera_id=frame_ctx.camera_id)

        return image

    def _draw_motion_mask(self, image: np.ndarray, polygon_pts: list) -> np.ndarray:
        """Rysuje ładny, półprzezroczysty poligon maski ruchu."""
        overlay = image.copy()
        pts = np.array(polygon_pts, np.int32).reshape((-1, 1, 2))
        
        # Wypełniamy poligon np. lekkim niebiesko-szarym kolorem (255, 200, 0) - BGR
        cv2.fillPoly(overlay, [pts], (255, 150, 0))
        
        # Łączymy warstwy z przezroczystością (alpha = 0.25)
        alpha = 0.25
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def _draw_tripwire_rules(self, image: np.ndarray, rules: list) -> None:
        """Rysuje linie wirtualne na klatce obrazu."""
        for rule in rules:
            if isinstance(rule, dict) and rule.get("type") == "tripwire":
                line = rule.get("line")
                if line and len(line) == 2:
                    p1, p2 = tuple(line[0]), tuple(line[1])
                    cv2.line(image, p1, p2, (0, 0, 255), box_thickness(image) + 1, lineType=cv2.LINE_AA)

                    rule_name = str(rule.get("name", "tripwire")).upper()
                    draw_label(image, rule_name, p1[0], p1[1], (0, 0, 255))

    def _get_color(self, detection_type: str) -> tuple[int, int, int]:
        try:
            if self.detector_provider is not None:
                return self.detector_provider.get_color(detection_type)
        except Exception:
            pass
        return BaseDetector.get_color()

    def _format_label(self, detection) -> str:
        label = detection.label or detection.detection_type
        text = f"{label}: {float(detection.confidence):.2f}"
        if hasattr(detection, "track_id") and detection.track_id is not None:
            text = f"ID:{detection.track_id} {text}"
        return text

    def _draw_audio_overlay(self, image: np.ndarray, camera_id: str) -> None:
        if self.audio_overlay_state is None:
            return

        info = self.audio_overlay_state.get(camera_id)

        if info is None:
            return

        import time

        now = time.time()
        age = now - info.updated_at
        text_age = now - info.last_text_at if info.last_text_at else 9999.0

        height, width = image.shape[:2]

        x = 20
        y = 35
        panel_width = 560
        panel_height = 120

        if age > 5.0:
            status = "AUDIO: stale"
        elif info.has_voice:
            status = f"AUDIO: voice {info.dbfs:.1f} dBFS"
        else:
            status = f"AUDIO: noise {info.dbfs:.1f} dBFS"

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (x, y),
            (x + panel_width, y + panel_height),
            (0, 0, 0),
            cv2.FILLED,
        )

        alpha = 0.55
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            image,
            status,
            (x + 15, y + 35),
            font,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        bar_x = x + 15
        bar_y = y + 55
        bar_w = panel_width - 30
        bar_h = 18

        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (80, 80, 80),
            cv2.FILLED,
        )

        level_ratio = self._dbfs_to_ratio(info.dbfs)
        fill_w = int(bar_w * level_ratio)

        bar_color = (0, 200, 0) if info.has_voice else (180, 180, 180)

        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            bar_color,
            cv2.FILLED,
        )

        if info.last_text and text_age <= 12.0:
            transcript = self._shorten_text(info.last_text, max_chars=58)
            transcript_text = f"SPEECH: {transcript}"

            cv2.putText(
                image,
                transcript_text,
                (x + 15, y + 98),
                font,
                0.72,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    def _dbfs_to_ratio(self, dbfs: float) -> float:
        # -60 dBFS = cisza, 0 dBFS = max
        value = (float(dbfs) + 60.0) / 60.0
        return max(0.0, min(1.0, value))

    def _shorten_text(self, text: str, max_chars: int = 60) -> str:
        text = " ".join(str(text).split())

        if len(text) <= max_chars:
            return text

        return text[: max_chars - 3] + "..."