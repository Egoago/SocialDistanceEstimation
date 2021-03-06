from typing import List
import numpy as np
from motpy import Detection, MultiObjectTracker
from src.detection.boundingbox import BoundingBox
from src.tracking.tracker import Tracker, BBoxFilter
from src.tracking.person import Person


class MotpyTracker(Tracker):
    def __init__(self, dt, bbox_filter: BBoxFilter):
        super().__init__(bbox_filter)
        model_spec = {
            'order_pos': 1, 'dim_pos': 2,  # position is a center in 2D space; under constant velocity model
            'order_size': 1, 'dim_size': 2,  # bounding box is 2 dimensional; under constant velocity model
            'q_var_pos': 1.0,  # process noise
            'q_var_size': 1.0,  # process noise
            'r_var_pos': 0.1,  # measurement noise
            'r_var_size': 0.1  # measurement noise
        }
        min_seconds_alive = 0.5
        min_steps_alive = int(min_seconds_alive * 1/dt)
        self.tracker = MultiObjectTracker(dt,
                                          model_spec=model_spec,
                                          active_tracks_kwargs={'min_steps_alive': min_steps_alive,
                                                                'max_staleness': 4},
                                          tracker_kwargs={'max_staleness': 4},
                                          matching_fn_kwargs={'min_iou': 0.01})

    def track(self, bboxes: List[BoundingBox]) -> List[Person]:
        bboxes = self.bbox_filter(bboxes)
        detections = []
        for bbox in bboxes:
            top_left, bottom_right = bbox.corners()
            detections.append(Detection(np.array([top_left[0],
                                                  top_left[1],
                                                  bottom_right[0],
                                                  bottom_right[1]], dtype=int).squeeze()))
        self.tracker.step(detections)
        tracks = self.tracker.active_tracks()
        people = []
        for track in tracks:
            people.append(Person(id=abs(hash(track.id)) % (10 ** 8),
                                 bbox=BoundingBox(int(track.box[0]),
                                                  int(track.box[1]),
                                                  int(track.box[2] - track.box[0]),
                                                  int(track.box[3] - track.box[1]))))
        # if len(tracks) > 0:
        #   print(step)
        # print('first track box: %s' % str(tracks[0].box))
        # print('first obsrv box: %s' % str(transform_bbox(self.dummy_bboxes[0])))
        # self.assertEqual(len(self.dummy_bboxes), len(tracks))
        # print('first track box: %s' % str(tracks[0].id))
        return people
