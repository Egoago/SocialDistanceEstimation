from .detector import Detector
from .boundingbox import BoundingBox


def create_detector() -> Detector:
    from .detectors import YoloV4
    from .detectors import TinyYoloV3
    return TinyYoloV3()
