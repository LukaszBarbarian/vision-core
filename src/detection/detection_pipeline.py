import time


class DetectionPipeline:
    def __init__(self, model, allowed_classes=None):
        self.model = model
        self.allowed_classes = allowed_classes

    async def run(self, frame):
        results = self.model(frame.frame_data, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]

                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if self.allowed_classes and name not in self.allowed_classes:
                    continue

                detections.append({
                    "camera_id": frame.camera_id,
                    "class": name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "timestamp": time.time()
                })

        return detections