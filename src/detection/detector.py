from typing import List
import numpy as np
from src.detection.boundingbox import BoundingBox


class Detector:
    def detect(self, image: np.array) -> List[BoundingBox]:
        raise NotImplementedError
