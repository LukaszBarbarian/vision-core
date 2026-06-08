import asyncio

from workers.camera_worker import CameraWorker


class CameraManager:
    def __init__(self, registry):
        self.workers = {
            cam_id: CameraWorker(cfg) 
            for cam_id, cfg in registry.items()
        }

    async def start_all(self):
        # Odpalamy każdego workera jako osobne zadanie asyncio
        tasks = [asyncio.create_task(w.run()) for w in self.workers.values()]
        await asyncio.gather(*tasks)