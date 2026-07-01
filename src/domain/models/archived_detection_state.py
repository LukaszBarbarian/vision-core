
from dataclasses import dataclass


@dataclass
class ArchivedDetectionState:
    scene_id: str
    boxes: list[dict]
    last_event_ts: float
    last_seen_ts: float