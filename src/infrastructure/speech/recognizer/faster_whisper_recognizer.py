# infrastructure/speech/recognizers/faster_whisper_recognizer.py

from __future__ import annotations

import numpy as np

from infrastructure.speech.recognizer.base import BaseSpeechRecognizer


class FasterWhisperSpeechRecognizer(BaseSpeechRecognizer):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "pl",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as ex:
            raise RuntimeError(
                "Missing dependency: faster-whisper. "
                "Install it with: pip install faster-whisper"
            ) from ex

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self.language = language
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        if sample_rate != 16000:
            raise ValueError(
                f"FasterWhisperSpeechRecognizer expects 16000 Hz audio, "
                f"got {sample_rate}"
            )

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return {
                "text": "",
                "confidence": None,
                "language": self.language,
            }

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        parts: list[str] = []
        confidences: list[float] = []

        for segment in segments:
            text = (segment.text or "").strip()

            if text:
                parts.append(text)

            # avg_logprob nie jest confidence 0..1,
            # ale można go później mapować na własną metrykę.
            if getattr(segment, "avg_logprob", None) is not None:
                confidences.append(float(segment.avg_logprob))

        text = " ".join(parts).strip()

        confidence = None
        if confidences:
            confidence = sum(confidences) / len(confidences)

        detected_language = getattr(info, "language", None) or self.language

        return {
            "text": text,
            "confidence": confidence,
            "language": detected_language,
        }