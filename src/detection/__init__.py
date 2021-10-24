from .detector import Detector
from .boundingbox import BoundingBox


def create_detector(use_gpu=None) -> Detector:
    """
    Creates a detector.

    :param use_gpu: Whether the GPU should be used, see Detector's documentation.
    :return: a Detector.
    """
    from .detectors import TinyYoloV3
    from .detectors import YoloV4

    # If changed, update the README (especially the section "Dependencies")
    return YoloV4(use_gpu)
