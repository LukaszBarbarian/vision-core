"""
Drawing helpers shared by the live stream renderer and the archive writer.

Sizes are defined for a 1280 px wide frame and scaled with the image, so
labels stay readable on 1440p frames and after the stream downsizes them.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
REFERENCE_WIDTH = 1280.0


def ui_scale(image: np.ndarray) -> float:
    return max(1.0, image.shape[1] / REFERENCE_WIDTH)


def box_thickness(image: np.ndarray) -> int:
    return max(2, round(2 * ui_scale(image)))


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Filled tag in the detector's colour, sitting on the top edge of a box."""
    height, width = image.shape[:2]
    scale = ui_scale(image)
    font_scale = 0.6 * scale
    thickness = max(1, round(1.6 * scale))
    pad = round(4 * scale)

    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    top = y - text_h - baseline - 2 * pad
    if top < 0:  # no room above the box: put the tag inside it
        top = y

    x2 = min(width - 1, x + text_w + 2 * pad)
    y2 = min(height - 1, top + text_h + baseline + 2 * pad)

    cv2.rectangle(image, (x, top), (x2, y2), color, cv2.FILLED)
    cv2.putText(
        image, text, (x + pad, top + pad + text_h),
        FONT, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA,
    )


def draw_info_bar(image: np.ndarray, text: str, corner: str = "top-right") -> None:
    """Semi-transparent dark pill with white text in a corner of the frame."""
    height, width = image.shape[:2]
    scale = ui_scale(image)
    font_scale = 0.6 * scale
    thickness = max(1, round(1.5 * scale))
    pad = round(10 * scale)
    margin = round(12 * scale)

    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    box_w, box_h = text_w + 2 * pad, text_h + baseline + 2 * pad

    if corner == "top-right":
        x1, y1 = width - box_w - margin, margin
    elif corner == "bottom-right":
        x1, y1 = width - box_w - margin, height - box_h - margin
    else:
        x1, y1 = margin, margin

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x1 + box_w), min(height - 1, y1 + box_h)

    roi = image[y1:y2, x1:x2]
    if roi.size:
        image[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)

    cv2.putText(
        image, text, (x1 + pad, y1 + pad + text_h),
        FONT, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
    )
