import json
import logging
from typing import Tuple, List

import cv2
import numpy as np

from src.detection import create_detector
from src.tracking import Person, create_tracker, BBoxFilter
from src.projection import create_calibrator, Intrinsics, opencv2opengl, project, opengl2opencv, back_project

from src.feedback import feedback as fb

logger = logging.getLogger(__name__)


class SocialDistanceEstimator:
    def __init__(self, dt: float, img_size: Tuple[int, int], **kwargs):
        self.settings = {
            'target_fps': None,  # TODO set value to desired
            # Draw a high-contrast disk at the center of the person's box
            'display_centers': True,
            # Draw a bounding box around the person
            'display_bounding_boxes': True,
            # Draw a circle in 3D around at each person's feet
            'display_proximity': True,
            # Amount of bounding boxes to gather before calibration
            'calibrate_at_bounding_boxes_count': 2000,
        }
        self.settings.update(**kwargs)

        logger.debug(f'Settings: {json.dumps(self.settings, indent=2)}')

        self.detector = create_detector(**self.settings)

        self.img_size = img_size
        bbox_filter = BBoxFilter(img_size=img_size,
                                 max_aspect=0.8,
                                 min_rel_height=0.1)
        self.tracker = create_tracker(dt, bbox_filter=bbox_filter)

        self.calibrator = create_calibrator(Intrinsics(res=np.array(img_size)), method='least_squares')

        self.p_bottom = np.zeros((self.settings['calibrate_at_bounding_boxes_count'], 2), dtype=float)
        self.p_top = np.zeros((self.settings['calibrate_at_bounding_boxes_count'], 2), dtype=float)
        self.bounding_boxes_count = 0
        self.camera = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        bounding_boxes = self.detector.detect(image)
        people = self.tracker.track(bounding_boxes)

        if self.bounding_boxes_count < self.settings['calibrate_at_bounding_boxes_count']:
            self.__calibrate(people)

        im = self.__create_image(image=image, people=people)
        return im

    def __create_image(self, image: np.ndarray, people: List[Person]) -> np.ndarray:
        return fb.feedback_image(self.camera, self.img_size, image, people, self.settings)

    def __calibrate(self, people: List[Person]):
        for person in people:
            self.p_top[self.bounding_boxes_count] = opencv2opengl(person.bbox.top(), self.img_size[1])
            self.p_bottom[self.bounding_boxes_count] = opencv2opengl(person.bbox.bottom(), self.img_size[1])
            self.bounding_boxes_count += 1
            if self.bounding_boxes_count == self.settings['calibrate_at_bounding_boxes_count']:
                # from src.projection.calibrators.test.drawing import draw_2d_points
                # ax = draw_2d_points(self.p_bottom, c='darkgreen', last=False)
                # draw_2d_points(self.p_top, c='darkred', ax=ax, res=self.img_size)
                self.camera = self.calibrator.calibrate(p_top=self.p_top, p_bottom=self.p_bottom)
                return
