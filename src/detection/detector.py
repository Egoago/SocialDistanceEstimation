from typing import List
import numpy as np

from .boundingbox import BoundingBox


class Detector:
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detects people in an image.

        :param image: An RGB image in a numpy ndarray with with values ranging from 0 to 255
        :return: A list containing a BoundingBox for each detected person
        """
        raise NotImplementedError
