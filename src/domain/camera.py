from pydantic import BaseModel, Field
from typing import List, Optional

class CameraMetadata(BaseModel):
    id: str = Field(..., description="Unique identifier for the camera")
    name: str = Field(..., description="Friendly display name")
    manufacturer: str = Field(..., description="Camera manufacturer, e.g., ezviz, hikvision")
    url: str = Field(..., description="Full RTSP connection URL")
    resolution: str = Field("1080p", description="Target stream resolution")
