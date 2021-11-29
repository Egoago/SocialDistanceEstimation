import json
import logging
from typing import Tuple, List

import numpy as np

from src.detection import create_detector
from src.imageprocessing import get_camera_params
from src.tracking import Person, create_tracker, BBoxFilter
from src.projection import create_calibrator, Intrinsics, opencv2opengl

from src.feedback import feedback_image

logger = logging.getLogger(__name__)


class SocialDistanceEstimator:
    def __init__(self, dt: float, input_shape: Tuple[int, int], output_shape: Tuple[int, int], **kwargs):
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

        self.input_shape = input_shape
        self.output_shape = output_shape
        bbox_filter = BBoxFilter(img_size=input_shape,
                                 max_aspect=0.8,
                                 min_rel_height=0.1)
        self.tracker = create_tracker(dt, bbox_filter=bbox_filter)

        self.camera_params = get_camera_params('files/images_calibration', 'jpg', 20, 8, 6)
        self.calibrator = create_calibrator(Intrinsics(res=np.array(input_shape)), method='least_squares')

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
        assert im.shape[:2] == self.output_shape
        return im

    def __create_image(self, image: np.ndarray, people: List[Person]) -> np.ndarray:
        return feedback_image(self.camera, self.input_shape, image, people, self.settings)

    def __calibrate(self, people: List[Person]):
        for person in people:
            self.p_top[self.bounding_boxes_count] = opencv2opengl(person.bbox.top(), self.input_shape[1])
            self.p_bottom[self.bounding_boxes_count] = opencv2opengl(person.bbox.bottom(), self.input_shape[1])
            self.bounding_boxes_count += 1
            if self.bounding_boxes_count == self.settings['calibrate_at_bounding_boxes_count']:
                # from src.projection.calibrators.test.drawing import draw_2d_points
                # ax = draw_2d_points(self.p_bottom, c='darkgreen', last=False)
                # draw_2d_points(self.p_top, c='darkred', ax=ax, res=self.input_shape)
                self.camera = self.calibrator.calibrate(p_top=self.p_top, p_bottom=self.p_bottom)
                del self.p_top, self.p_bottom
                return
