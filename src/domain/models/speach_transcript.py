from dataclasses import dataclass


@dataclass(slots=True)
class SpeechTranscript:
    camera_id: str
    text: str
    timestamp_start: float
    timestamp_end: float
    confidence: float | None = None
    language: str | None = None
    stream_session_id: str | None = None