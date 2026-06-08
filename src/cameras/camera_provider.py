from core.settings import Settings
from domain.camera_configuration import CameraConfiguration


class CameraProvider:
    @staticmethod
    def get_active_cameras(settings: Settings | None = None) -> list[CameraConfiguration]:
        return [cam for cam in settings.registry.values() if cam.active]