# infrastructure/speech/recognizers/dummy.py

from __future__ import annotations

import numpy as np

from infrastructure.speech.recognizer.base import BaseSpeechRecognizer

class DummySpeechRecognizer(BaseSpeechRecognizer):
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        return {
            "text": "",
            "confidence": None,
            "language": "pl",
        }