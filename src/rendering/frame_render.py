import cv2
import numpy as np

from cameras.frame_context import FrameContext
from detection.detectors.base import BaseDetector


class FrameRenderer:
    def __init__(self, detector_provider=None):
        self.detector_provider = detector_provider

    def render(
        self,
        frame_ctx: FrameContext,
        visible_types: set[str] | None = None,
    ) -> np.ndarray:
        image = frame_ctx.frame_data.copy()
        detections = getattr(frame_ctx, "detections", []) or []

        height, width = image.shape[:2]

        visible_types = visible_types or set()

        drawn = 0

        for detection in detections:
            detection_type = str(detection.detection_type)

            if visible_types and detection_type not in visible_types:
                continue

            color = self._get_color(detection_type)

            x1 = max(0, min(width - 1, int(detection.x1)))
            y1 = max(0, min(height - 1, int(detection.y1)))
            x2 = max(0, min(width - 1, int(detection.x2)))
            y2 = max(0, min(height - 1, int(detection.y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            label = self._format_label(detection)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(image, (cx, cy), 4, color, -1)

            self._draw_label(
                image=image,
                text=label,
                x=x1,
                y=y1,
                color=color,
            )

            drawn += 1

        self._draw_debug_header(
            image=image,
            frame_ctx=frame_ctx,
            detections_count=len(detections),
            drawn_count=drawn,
        )

        return image

    def _get_color(self, detection_type: str) -> tuple[int, int, int]:
        try:
            if self.detector_provider is not None:
                return self.detector_provider.get_color(detection_type)
        except Exception:
            pass

        return BaseDetector.get_color()

    def _format_label(self, detection) -> str:
        label = detection.label or detection.detection_type
        return f"{label}: {float(detection.confidence):.2f}"

    def _draw_label(
        self,
        image,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        height, width = image.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        text_x = x
        text_y = max(text_height + baseline + 8, y - 6)

        bg_x1 = text_x
        bg_y1 = max(0, text_y - text_height - baseline - 6)
        bg_x2 = min(width - 1, text_x + text_width + 8)
        bg_y2 = min(height - 1, text_y + baseline)

        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            cv2.FILLED,
        )

        cv2.putText(
            image,
            text,
            (text_x + 4, text_y - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_debug_header(
        self,
        image,
        frame_ctx: FrameContext,
        detections_count: int,
        drawn_count: int,
    ) -> None:
        debug_text = (
            f"{frame_ctx.camera_id} | frame={frame_ctx.frame_id} | "
            f"scene={getattr(frame_ctx, 'scene_id', None)} | "
            f"detections={detections_count} | drawn={drawn_count}"
        )

        cv2.putText(
            image,
            debug_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
