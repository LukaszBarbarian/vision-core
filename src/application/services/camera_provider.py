from domain.models.camera_configuration import CameraConfiguration
from infrastructure.config.settings import Settings

class CameraProvider:
    @staticmethod
    def get_active_cameras(settings: Settings | None = None) -> list[CameraConfiguration]:
        return [cam for cam in settings.registry.values() if cam.camera.active]