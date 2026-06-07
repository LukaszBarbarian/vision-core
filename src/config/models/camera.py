from pydantic import BaseModel, Field
from typing import List, Optional

class CameraMetadata(BaseModel):
    id: str = Field(..., description="Identyfikator, np. 2 lub cam_kurnik")
    name: str
    manufacturer: str
    url: str
    resolution: str

class DetectionSpecification(BaseModel):
    detection_type: str = Field(..., description="Typ detekcji, np. 'poultry', 'face'")
    ai_fps: int = Field(3, description="Klatki na sekundę specyficzne dla tej detekcji")

class CameraConfiguration(BaseModel):
    camera: CameraMetadata
    active: bool = True
    detections: List[DetectionSpecification] = Field(default_factory=list)
    archive: bool = Field(True, description="Czy zapisywać wyniki do plików analitycznych (Parquet)")