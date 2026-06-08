from pydantic import BaseModel, Field
import datetime

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionEvent(BaseModel):
    camera_id: str
    detection_type: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc))
