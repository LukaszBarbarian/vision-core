import asyncio
from typing import Dict, List, Any, TypeVar, Generic

# Typ generyczny, żeby kod był czytelny dla IDE
T = TypeVar('T')

class BaseBus(Generic[T]):
    def __init__(self, queue_size: int):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._queue_size = queue_size

    def subscribe(self, topic: str) -> asyncio.Queue:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        q = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[topic].append(q)
        return q

    async def publish(self, topic: str, data: T, drop_oldest: bool = True):
        if topic not in self._subscribers:
            return

        for q in self._subscribers[topic]:
            if q.full():
                if drop_oldest:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                else:
                    # Jeśli nie chcemy gubić danych, po prostu pomijamy 
                    # (lub możemy rzucić wyjątek/log)
                    continue
            await q.put(data)

class FrameBus(BaseBus):
    def __init__(self):
        # Szyna wideo: maxsize=1, zawsze zrzucamy starą klatkę
        super().__init__(queue_size=1)

    async def publish(self, topic: str, frame: Any):
        await super().publish(topic, frame, drop_oldest=True)

class EventBus(BaseBus):
    def __init__(self):
        # Szyna zdarzeń: większy bufor, żeby nie przegapić detekcji
        super().__init__(queue_size=100)

    async def publish(self, topic: str, event: Any):
        await super().publish(topic, event, drop_oldest=False)

# Instancje globalne
frame_bus = FrameBus()
event_bus = EventBus()