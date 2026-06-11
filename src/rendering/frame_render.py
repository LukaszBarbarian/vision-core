import cv2
import numpy as np
from cameras.frame_context import FrameContext

class FrameRenderer:
    def render(
        self,
        frame_ctx: FrameContext,
        visible_types: set[str] | None = None
    ) -> np.ndarray:

        image = frame_ctx.frame_data.copy()
        
        # Bezpieczne pobranie detekcji
        detections = getattr(frame_ctx, 'detections', [])
        
        for detection in detections:
            if visible_types and detection.detection_type not in visible_types:
                continue

            # Rysowanie ramki
            cv2.rectangle(
                image,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 255, 0),
                2
            )

            label = f"{detection.label or detection.detection_type}: {detection.confidence:.2f}"
            
            # Parametry tekstu
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0  # Powiększone
            thickness = 2
            
            # Obliczenie rozmiaru tekstu, aby stworzyć tło
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Pozycja tła (prostokąt nad ramką)
            x, y = detection.x1, detection.y1
            
            # Rysowanie tła pod tekst (dla czytelności)
            cv2.rectangle(
                image, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                (0, 255, 0), 
                cv2.FILLED
            )
            
            # Rysowanie napisu (kolor czarny dla kontrastu na zielonym tle)
            cv2.putText(
                image,
                label,
                (x, y - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA
            )

        return image