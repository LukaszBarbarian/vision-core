# domain/models/camera.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class CameraMetadata(BaseModel):
    id: str
    name: str
    manufacturer: str
    url: str
    resolution: str
    # Te pola w YAML są wewnątrz bloku "camera"
    active: bool = True
    scene_tracking: bool = False
    audio_enabled: bool = True
    motion_mask_polygon: List[List[int]] = Field(default_factory=list)
    rules: List[Dict[str, Any]] = Field(default_factory=list)