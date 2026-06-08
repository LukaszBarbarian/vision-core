from pydantic import BaseModel, Field
from typing import List
from .camera import CameraMetadata

class DetectionSpecification(BaseModel):
    detection_type: str = Field(..., description="Type of AI model to apply, e.g., poultry, face")
    ai_fps: int = Field(3, description="Sampling rate (frames per second) for this specific model")

class CameraConfiguration(BaseModel):
    camera: CameraMetadata
    active: bool = Field(True, description="Master flag to enable or disable this camera pipeline")
    detections: List[DetectionSpecification] = Field(default_factory=list, description="List of active AI analytics configurations")
    archive: bool = Field(True, description="Flag to control telemetry data logging to historical files")