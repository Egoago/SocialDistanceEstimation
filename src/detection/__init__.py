from .detector import Detector
from .boundingbox import BoundingBox


def create_detector() -> Detector:
    from .detectors import TinyYoloV3
    from .detectors import YoloV4

    return YoloV4()
