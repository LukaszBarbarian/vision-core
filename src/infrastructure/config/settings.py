import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

from domain.models.camera_configuration import CameraConfiguration
from domain.models.detection import DetectionSpecification


class Settings:
    def __init__(self, config_path: Path | None = None):

        base_dir = Path(__file__).resolve().parent
        self.config_path = config_path or (base_dir / "nvr_registry.yaml")

        self.registry = {}
        self.detection_configs = {}

        load_dotenv()
        self.load_registry()

    def load_registry(self):
        raw_content = self.config_path.read_text(encoding="utf-8")
        expanded = os.path.expandvars(raw_content)

        raw_data = yaml.safe_load(expanded) or {}

        for item in raw_data.get("cameras", []):

            config = CameraConfiguration(**item)
            self.registry[config.camera.id] = config

            self.detection_configs[config.camera.id] = [
                DetectionSpecification(**d)
                for d in item.get("detections", [])
            ]

