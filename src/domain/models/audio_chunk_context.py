# domain/models/audio_context.py
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class AudioChunkContext:
    camera_id: str
    chunk_id: int
    pcm: np.ndarray              # int16 mono, shape: (samples,)
    sample_rate: int             # np. 16000
    timestamp_start: float
    timestamp_end: float
    stream_session_id: str | None = None