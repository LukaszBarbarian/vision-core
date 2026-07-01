import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SceneState:
    scene_id: str
    scene_no: int
    reference: np.ndarray
    last_check_ts: float
    scene_started_ts: float
    consecutive_changes: int = 0
    last_debug_ts: float = 0.0
