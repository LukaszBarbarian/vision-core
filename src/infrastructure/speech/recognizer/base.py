# infrastructure/speech/recognizers/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseSpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """
        audio: float32 mono [-1.0, 1.0]
        sample_rate: np. 16000

        Return:
        {
            "text": "...",
            "confidence": float | None,
            "language": "pl" | None,
        }
        """
        raise NotImplementedError