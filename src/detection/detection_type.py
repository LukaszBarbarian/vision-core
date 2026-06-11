from enum import Enum

class DetectionType(str, Enum):
    CHICKEN = "chicken_detection"
    CAR = "car_detection"
    LICENSE_PLATE = "anpr_detection"
    PERSON = "person_detection"
    FACE = "face_detection"