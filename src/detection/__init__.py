from .detector import Detector
from .boundingbox import BoundingBox


def create_detector(target_fps=None, use_gpu=None, detector_name=None, **kwargs) -> Detector:
    """
    Creates a detector.

    :param detector_name: The name of the Detector which will be used. Otherwise, a ValueError is raised.
    :param target_fps: The FPS value to be targeted when choosing the appropriate detector.
    :param use_gpu: Whether the GPU should be used, see Detector's documentation.
    :return: a Detector.
    """
    from .detectors import TinyYoloV3
    from .detectors import YoloV4
    from .detectors import CompositeDetector

    if detector_name is not None:
        if detector_name == 'TinyYoloV3':
            return TinyYoloV3(use_gpu)
        if detector_name == 'YoloV4':
            return YoloV4(use_gpu)
        if detector_name == 'CompositeDetector':
            return CompositeDetector(use_gpu)
        raise ValueError(f'{detector_name} does not exist')

    if target_fps is not None:
        if target_fps < 5:
            return YoloV4(use_gpu)
        if target_fps > 20:
            return TinyYoloV3(use_gpu)

    # If changed, update the README (especially the section "Dependencies")
    return YoloV4(use_gpu)
