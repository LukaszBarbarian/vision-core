import re
import cv2
import numpy as np
import easyocr

from infrastructure.ai_detection.detectors.base import BaseDetector
from infrastructure.ai_detection.providers.model_provider import DetectionModelProvider
from domain.models.detection_type import DetectionType

from domain.models.detection_result import DetectionResult


class AnprDetector(BaseDetector):

    def __init__(
        self,
        model_provider: DetectionModelProvider
    ):
        self._car_model = model_provider.get_model(
            DetectionType.CAR
        )

        self._plate_model = model_provider.get_model(
            DetectionType.LICENSE_PLATE
        )

        self._reader = easyocr.Reader(
            ['pl', 'en'],
            gpu=True
        )

        self._vehicle_classes = [
            2, 3, 5, 7
        ]

    def _clean(self, text: str) -> str:
        return re.sub(
            r'[^A-Z0-9]',
            '',
            text.upper().strip()
        )

    def _valid(self, text: str) -> bool:

        length = len(text)

        if length == 8:

            suffix = text[3:]

            valid_patterns = [
                r'^\d{5}$',
                r'^\d{4}[A-Z]$',
                r'^\d{3}[A-Z]{2}$',
                r'^\d[A-Z]\d{3}$',
                r'^\d{2}[A-Z]\d{2}$',
                r'^\d[A-Z]{2}\d{2}$',
                r'^\d{2}[A-Z]{2}\d$'
            ]

            return any(
                re.match(p, suffix)
                for p in valid_patterns
            )

        elif length == 7:

            suffix_5 = text[2:]
            suffix_4 = text[3:]

            patterns_5 = [
                r'^\d{5}$',
                r'^\d{4}[A-Z]$',
                r'^\d{3}[A-Z]{2}$',
                r'^[A-Z]\d{3}[A-Z]$'
            ]

            patterns_4 = [
                r'^\d{4}$',
                r'^\d{3}[A-Z]$',
                r'^\d{2}[A-Z]{2}$',
                r'^[A-Z]\d{3}$'
            ]

            return (
                any(
                    re.match(p, suffix_5)
                    for p in patterns_5
                )
                or
                any(
                    re.match(p, suffix_4)
                    for p in patterns_4
                )
            )

        return False

    def execute(
        self,
        frame_data: np.ndarray,
        context: dict
    ) -> list[DetectionResult]:

        if frame_data is None or frame_data.size == 0:
            return []

        car_boxes = context.get(
            "detected_cars"
        )

        if car_boxes is None:

            results = self._car_model(
                frame_data,
                classes=self._vehicle_classes,
                conf=0.3,
                imgsz=1280,
                verbose=False
            )

            car_boxes = (
                results[0].boxes
                if (
                    results
                    and len(results[0].boxes) > 0
                )
                else []
            )

        if "detected_cars" in context:
            car_boxes = context["detected_cars"]
        else:
            results = self._car_model(...)
            car_boxes = results[0].boxes if results and len(results[0].boxes) > 0 else []
            context["detected_cars"] = car_boxes

        if car_boxes is None or len(car_boxes) == 0:
            return []

        clean_source = context.get(
            "clean_frame",
            frame_data
        )

        detections = []

        for car_box in car_boxes:

            cx1, cy1, cx2, cy2 = map(
                int,
                car_box.xyxy[0]
            )

            cx1 = max(0, cx1)
            cy1 = max(0, cy1)

            cx2 = min(
                clean_source.shape[1],
                cx2
            )

            cy2 = min(
                clean_source.shape[0],
                cy2
            )

            car_crop = clean_source[
                cy1:cy2,
                cx1:cx2
            ]

            if car_crop.size == 0:
                continue

            plate_results = self._plate_model(
                car_crop,
                conf=0.2,
                verbose=False
            )

            if (
                not plate_results
                or len(
                    plate_results[0].boxes
                ) == 0
            ):
                continue

            for plate_box in plate_results[0].boxes:

                px1, py1, px2, py2 = map(
                    int,
                    plate_box.xyxy[0]
                )

                plate_crop = car_crop[
                    py1:py2,
                    px1:px2
                ]

                if plate_crop.size == 0:
                    continue

                plate_h, plate_w = plate_crop.shape[:2]

                if plate_w < 40 or plate_h < 12:
                    continue

                aspect_ratio = plate_w / max(1, plate_h)

                if aspect_ratio < 2.0 or aspect_ratio > 6.5:
                    continue

                target_h = 120

                scale = (
                    target_h
                    / plate_crop.shape[0]
                )

                plate_resized = cv2.resize(
                    plate_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

                gray = cv2.cvtColor(
                    plate_resized,
                    cv2.COLOR_BGR2GRAY
                )

                clahe = cv2.createCLAHE(
                    clipLimit=3.0,
                    tileGridSize=(8, 8)
                )

                gray = clahe.apply(gray)

                processed_plate = (
                    cv2.bilateralFilter(
                        gray,
                        d=9,
                        sigmaColor=75,
                        sigmaSpace=75
                    )
                )

                ocr_results = self._reader.readtext(
                    processed_plate,
                    detail=1,
                    paragraph=False,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    width_ths=0.7
                )

                if not ocr_results:
                    continue

                ocr_results_sorted = sorted(
                    ocr_results,
                    key=lambda r: r[0][0][0]
                )

                combined_text = ""
                conf_scores = []

                for (
                    bbox,
                    text,
                    conf
                ) in ocr_results_sorted:

                    if conf < 0.15:
                        continue

                    cleaned = self._clean(text)

                    if cleaned:
                        combined_text += cleaned
                        conf_scores.append(conf)

                best_text = None
                best_conf = 0.0

                if self._valid(combined_text):

                    best_text = combined_text

                    best_conf = (
                        sum(conf_scores)
                        / len(conf_scores)
                        if conf_scores
                        else 0.0
                    )

                else:

                    for (
                        bbox,
                        text,
                        conf
                    ) in ocr_results_sorted:

                        cleaned = self._clean(
                            text
                        )

                        if (
                            conf >= 0.3
                            and self._valid(
                                cleaned
                            )
                        ):
                            if conf > best_conf:
                                best_conf = conf
                                best_text = cleaned

                if best_text is None:
                    continue

                global_x1 = cx1 + px1
                global_y1 = cy1 + py1
                global_x2 = cx1 + px2
                global_y2 = cy1 + py2

                detections.append(
                    DetectionResult(
                        detection_type="anpr_detection",
                        x1=global_x1,
                        y1=global_y1,
                        x2=global_x2,
                        y2=global_y2,
                        confidence=best_conf,
                        label=best_text
                    )
                )

        return detections
    



    @classmethod
    def get_color(cls) -> tuple[int, int, int]:
        return (255, 255, 0)  # cyan BGR

    @classmethod
    def get_display_name(cls) -> str:
        return "Tablica"