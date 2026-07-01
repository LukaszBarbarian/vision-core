# domain/models/camera_configuration.py
from typing import List
from pydantic import BaseModel, Field
from domain.models.camera import CameraMetadata
from domain.models.detection import DetectionSpecification

class CameraConfiguration(BaseModel):
    camera: CameraMetadata
    detections: List[DetectionSpecification] = Field(default_factory=list)
    archive: bool = Field(True)