from typing import List

from pydantic import BaseModel, Field

from domain.camera import CameraMetadata
from domain.detection import DetectionSpecification


class CameraConfiguration(BaseModel):
    camera: CameraMetadata
    active: bool = Field(
        True,
        description="Master flag to enable or disable this camera pipeline",
    )
    detections: List[DetectionSpecification] = Field(
        default_factory=list,
        description="List of active AI analytics configurations",
    )
    archive: bool = Field(
        True,
        description="Flag to control telemetry data logging to historical files",
    )
    scene_tracking: bool = Field(
        False,
        description=(
            "Enable visual scene-change tracking for PTZ cameras. "
            "Keep False for static cameras to avoid unnecessary scene references."
        ),
    )
