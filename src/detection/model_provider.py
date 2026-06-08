from ultralytics import YOLO

class ModelProvider:
    def __init__(self):
        self._models = {}

    def get_model(self, model_name: str = "yolov8n.pt") -> YOLO:
        if model_name not in self._models:
            print(f"[ModelProvider] Loading weights into memory: {model_name}")
            # Tu ładujemy czysty model bez żadnych filtrów klas
            self._models[model_name] = YOLO(model_name)
        return self._models[model_name]