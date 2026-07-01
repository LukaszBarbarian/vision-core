from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FaceDetectorOptions:
    require_person: bool = False
    fallback_to_full_frame: bool = True
    use_upper_body_roi: bool = True
    model: str = "hog"
    number_of_times_to_upsample: int = 1
    crop_upscale: float = 1.4
    full_frame_scale: float = 0.75
    min_face_size: int = 18
    num_jitters: int = 1
    debug: bool = False