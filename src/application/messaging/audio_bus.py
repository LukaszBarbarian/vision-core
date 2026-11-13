from typing import Any

from application.messaging.bus import BaseBus

class AudioBus(BaseBus):
    """
    Audio is a continuous signal: unlike video, a dropped chunk leaves a hole
    in the speech window. Each subscriber may fall behind by up to
    queue_size chunks (30 s of 1 s chunks) while the recogniser is busy;
    only beyond that are the oldest chunks dropped.
    """

    def __init__(self, queue_size: int = 30):
        super().__init__(queue_size=queue_size)

    async def publish(self, topic: str, frame: Any):
        await super().publish(topic, frame)