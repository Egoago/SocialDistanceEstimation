from person import Person
from tracker import Tracker


def get_tracker() -> Tracker:
    from tracking.trackers.motpyTracker import MotpyTracker
    return MotpyTracker(dt=30)  # TODO insert td if known
