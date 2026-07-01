import cv2
import numpy as np


class ZoneFilter:
    def __init__(self, polygon_roi: list[tuple[int, int]] | None = None):
        self.polygon_roi = polygon_roi

    def create_mask(self, frame_shape: tuple[int, ...]) -> np.ndarray | None:
        """Zwraca maskę wycinającą wszystko poza zdefiniowanym poligonem, lub None jeśli brak."""
        if not self.polygon_roi:
            return None

        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        poly = np.array(self.polygon_roi, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        
        return mask

    @staticmethod
    def is_line_crossed(p1: tuple[int, int], p2: tuple[int, int], obj_point: tuple[int, int]) -> float:
        """
        Zwraca wartość krzyżową do ustalenia strony linii. 
        Wymaga p1 i p2 (punkty linii) oraz obj_point (np. środek dolnej krawędzi bounding boxa).
        """
        return (obj_point[0] - p1[0]) * (p2[1] - p1[1]) - (obj_point[1] - p1[1]) * (p2[0] - p1[0])