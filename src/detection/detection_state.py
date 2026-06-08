class DetectionState:
    def __init__(self):
        self.latest = {}  # camera_id -> list[DetectionEvent]

    def update(self, event):
        self.latest.setdefault(event.camera_id, [])
        self.latest[event.camera_id].append(event)

        # limit pamięci
        self.latest[event.camera_id] = self.latest[event.camera_id][-50:]