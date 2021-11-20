from abc import abstractmethod, ABCMeta
from typing import List
import numpy as np

from .boundingbox import BoundingBox


class Detector(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, use_gpu=None):
        """
        Creates a detector.

        :param use_gpu: If use_gpu is True, the detector will utilize the GPU if a GPU is available, otherwise
                raise an Exception.
            If False, only CPU will be used.
            If None, the default behaviour is chosen: use GPU if possible, otherwise use CPU.
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detects people in an image.

        :param image: An RGB image in a numpy ndarray with with values ranging from 0 to 255
        :return: A list containing a BoundingBox for each detected person (unknown order)
        """
        raise NotImplementedError
