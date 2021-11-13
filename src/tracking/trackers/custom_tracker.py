from typing import List

from src.detection import BoundingBox
from src.tracking.person import Person
from src.tracking.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self):
        self.people = []

    def track(self, bboxes: List[BoundingBox]) -> List[Person]:
        raise NotImplementedError
