from .detector import Detector
from .boundingbox import BoundingBox


def create_detector(target_fps=None, use_gpu=None, **kwargs) -> Detector:
    """
    Creates a detector.

    :param target_fps: The FPS value to be targeted when choosing the appropriate detector.
    :param use_gpu: Whether the GPU should be used, see Detector's documentation.
    :return: a Detector.
    """
    from .detectors import TinyYoloV3
    from .detectors import YoloV4

    if target_fps is not None:
        if target_fps < 5:
            # TODO possible force gpu on
            return YoloV4(use_gpu)
        if target_fps > 20:
            # TODO possible force gpu off
            return TinyYoloV3(use_gpu)

    # If changed, update the README (especially the section "Dependencies")
    return YoloV4(use_gpu)
