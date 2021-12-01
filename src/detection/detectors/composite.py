from typing import List

import numpy as np

from src.detection import Detector, BoundingBox
from src.detection.detectors import YoloV4, TinyYoloV3


class CompositeDetector(Detector):
    def __init__(self, use_gpu=None, models=None):
        super().__init__(use_gpu)
        self.models = models if models is not None else [YoloV4(use_gpu), TinyYoloV3(use_gpu)]
        self.iteration = 0

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        active = self.models[self.iteration % len(self.models)]
        self.iteration += 1
        return active.detect(image)
