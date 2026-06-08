from pydantic import BaseModel, Field
import datetime

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionEvent(BaseModel):
    camera_id: str
    detection_type: str
    confidence: float
    bbox: BoundingBox
    # Automatycznie wbija aktualny czas przy tworzeniu obiektu
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)