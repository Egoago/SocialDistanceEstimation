from person import Person
from tracker import Tracker, BBoxFilter


def create_tracker(dt: float, bbox_filter: BBoxFilter) -> Tracker:
    from .trackers.motpyTracker import MotpyTracker
    return MotpyTracker(dt=dt, bbox_filter=bbox_filter)
