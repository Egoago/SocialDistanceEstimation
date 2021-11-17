from typing import List, NamedTuple, Tuple
from src.detection.boundingbox import BoundingBox
from src.tracking.person import Person


class BBoxFilter(NamedTuple):
    img_size: Tuple[int, int]
    min_aspect: float = 0
    max_aspect: float = 3
    min_rel_height: float = 0
    max_rel_height: float = 1
    min_rel_width: float = 0
    max_rel_width: float = 1

    def __call__(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        return list(filter(lambda bbox: self.__filter_func__(bbox), bboxes))

    def __filter_func__(self, bbox: BoundingBox) -> bool:
        aspect = bbox.w / bbox.h
        rel_width = bbox.w / self.img_size[0]
        rel_height = bbox.h / self.img_size[1]
        return self.min_aspect < aspect < self.max_aspect and \
               self.min_rel_width < rel_width < self.max_rel_width and \
               self.min_rel_height < rel_height < self.max_rel_height


class Tracker:
    def __init__(self, bbox_filter: BBoxFilter):
        self.bbox_filter = bbox_filter

    def track(self, bboxes: List[BoundingBox]) -> List[Person]:
        raise NotImplementedError
