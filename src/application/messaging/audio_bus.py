from typing import Any

from application.messaging.bus import BaseBus

class AudioBus(BaseBus):
    def __init__(self):
        super().__init__(queue_size=1)

    async def publish(self, topic: str, frame: Any):
        await super().publish(topic, frame)