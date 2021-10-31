import numpy as np
from src.detection.boundingbox import BoundingBox


class Person:
    def __init__(self, id: int, bbox: BoundingBox):
        rng = np.random.default_rng(seed=id)
        self.color = rng.integers(0, 200, 3).tolist()
        self.bbox = bbox
