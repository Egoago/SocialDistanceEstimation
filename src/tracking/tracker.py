from typing import List
from src.detection.boundingbox import BoundingBox
from src.tracking.Person import Person
import numpy as np


class Tracker:
    def track(self, bboxes: List[BoundingBox]) -> List[Person]:
        raise NotImplementedError

    def transform_bbox(bbox: BoundingBox):
        return np.array([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h], dtype=int).squeeze()

