from typing import List
from pydantic import BaseModel, Field

class DetectionSpecification(BaseModel):
    detection_type: str = Field(..., description="Type of detection pipeline, e.g. poultry, face")
    ai_fps: int = Field(3, description="How many frames per second this detection should run")
    classes: List[str] = []