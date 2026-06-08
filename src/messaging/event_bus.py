
from typing import Any
from messaging.bus import BaseBus


class EventBus(BaseBus):
    def __init__(self):
        # Szyna zdarzeń: większy bufor, żeby nie przegapić detekcji
        super().__init__(queue_size=100)

    async def publish(self, topic: str, event: Any):
        await super().publish(topic, event)