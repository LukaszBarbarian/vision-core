import asyncio
import time

import numpy as np

from application.messaging.audio_bus import AudioBus
from application.messaging.event_bus import EventBus
from domain.models.audio_chunk_context import AudioChunkContext
from infrastructure.speech.speach_worker import SpeechWorker

RATE = 16000
CHUNK_SECONDS = 0.1


class SlowRecognizer:
    """Takes longer than the chunks arrive, like Whisper on a busy CPU."""

    def __init__(self):
        self.window_lengths = []

    def transcribe(self, audio, sample_rate):
        self.window_lengths.append(len(audio))
        time.sleep(0.15)
        return {"text": ""}


def chunk(i):
    return AudioChunkContext(
        camera_id="cam",
        chunk_id=i,
        pcm=np.zeros(int(RATE * CHUNK_SECONDS), dtype=np.int16),
        sample_rate=RATE,
        timestamp_start=i * CHUNK_SECONDS,
        timestamp_end=(i + 1) * CHUNK_SECONDS,
    )


def test_slow_recognizer_still_gets_a_gap_free_window():
    async def scenario():
        audio_bus = AudioBus()
        recognizer = SlowRecognizer()
        worker = SpeechWorker(
            camera_id="cam",
            audio_bus=audio_bus,
            event_bus=EventBus(),
            recognizer=recognizer,
            window_seconds=4.0,
            step_seconds=0.0,
            min_audio_seconds=0.0,
        )
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.05)

        for i in range(40):  # 4 s of audio, published faster than it is transcribed
            await audio_bus.publish("cam", chunk(i))
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.6)
        worker.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return recognizer.window_lengths

    lengths = asyncio.run(scenario())

    # the last window holds all 40 chunks: nothing was dropped on the way
    assert lengths[-1] == 40 * int(RATE * CHUNK_SECONDS)
    # and the worker caught up instead of transcribing one chunk at a time
    assert len(lengths) < 40
