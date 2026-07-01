import asyncio
from typing import Dict, List, Any, TypeVar, Generic

T = TypeVar("T")


class BaseBus(Generic[T]):
    def __init__(self, queue_size: int):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._queue_size = queue_size

    def subscribe(self, topic: str) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=self._queue_size)

        if topic not in self._subscribers:
            self._subscribers[topic] = []

        self._subscribers[topic].append(q)
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(queue)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]
            except ValueError:
                pass

    async def publish(self, topic: str, data: T):
        subscribers = self._subscribers.get(topic)
        if not subscribers:
            return

        for q in subscribers:
            try:
                # 👉 NON-BLOCKING, ALWAYS LATEST FRAME
                q.put_nowait(data)

            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    # jeśli nadal full → drop frame (lepiej niż blokada)
                    pass