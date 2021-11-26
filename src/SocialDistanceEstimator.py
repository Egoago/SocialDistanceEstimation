import json
import logging
import warnings
from typing import Tuple, List

import cv2
import numpy as np

from src.detection import create_detector
from src.imageprocessing import get_camera_params
from src.tracking import Person, create_tracker, BBoxFilter
from src.projection import create_calibrator, Intrinsics, opencv2opengl, project, opengl2opencv, back_project
from src.distances import distance_calc

logger = logging.getLogger(__name__)


class SocialDistanceEstimator:
    def __init__(self, dt: float, img_size: Tuple[int, int], **kwargs):
        self.settings = {
            'target_fps': 30,  # TODO set value to desired
            # Draw a high-contrast disk at the center of the person's box
            'display_centers': False,
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

        self.camera_params = get_camera_params('files/images_calibration','jpg',20,8,6)
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
        # TODO assert returning image size is equal to image size expected by video writer in main
        return im

    def __create_image(self, image: np.ndarray, people: List[Person]) -> np.ndarray:
        # TODO move to feedback package
        centerp = []
        bbs = []
        for person in people:
            if self.camera is not None:
                center = back_project(np.array(opencv2opengl(person.bbox.bottom(), self.img_size[1])),
                                      self.camera)
                centerp.append(center)
                bbs.append(person.bbox)

            if self.camera is not None and self.settings.get('display_proximity'):
                res = 20
                radius = 1000
                center = back_project(np.array(opencv2opengl(person.bbox.bottom(), self.img_size[1])),
                                      self.camera)
                centerp.append(center)
                pixels = []
                for i in np.linspace(0, 2 * np.pi.real, res):
                    point = center + np.array([np.cos(i), 0, np.sin(i)], dtype=float) * radius
                    pixel = opengl2opencv(tuple(project(point, self.camera)[0]), self.img_size[1])
                    pixels.append(pixel)
                cv2.polylines(image, np.int32([pixels]), True, (255, 128, 0), 2)

            if self.settings.get('display_centers'):
                center = person.bbox.x + person.bbox.w // 2, person.bbox.y + person.bbox.h // 2
                cv2.circle(image, center, 6, (0, 255, 0), 8)
                cv2.circle(image, center, 4, (255, 0, 255), 4)

        dist = 150
        locations = distance_calc(centerp, dist)
        # TODO locations are broken, lines are not drawn

        risky = locations[0]
        critic = locations[1]
        idx = locations[2]
        red = (0, 0, 255)
        orange = (0, 165, 255)
        green = (0, 255, 0)

        bew_img = np.zeros((480, 360, 3), np.uint8)

        frame_width = image.shape[1]
        sf, sfx, sfy = self.scaling(centerp, frame_width)
        for r in risky:
            cv2.line(bew_img, (int(r[0] * sfx), int(r[1] * sfy)), (int(r[2] * sfx), int(r[3] * sfy)), orange, 2)

        for c in critic:
            cv2.line(bew_img, (int(c[0] * sfx), int(c[1] * sfy)), (int(c[2] * sfx), int(c[3] * sfy)), red, 2)

        for cp in centerp:
            cv2.circle(bew_img, (int(cp[0] * sfx), int(cp[1] * sfy)), 4, (255, 255, 255), -1)

        if sf is None:
            # TODO
            warnings.warn('Handle sf is None, setting to 1')
            sf = 1
        image = cv2.resize(image, (int(image.shape[1] * sf), 720))

        for b in range(len(bbs)):
            people[b].color = green
            if b not in idx:
                people[b].color = red

            cv2.rectangle(image, (int(bbs[b].x * sf), int(bbs[b].y * sf)),
                          ((int(bbs[b].x * sf) + int(bbs[b].w * sf)),
                           (int(bbs[b].y * sf) + int(bbs[b].h * sf))), people[b].color)

        text = np.zeros((240, 360, 3), np.uint8)
        txt1 = 'Keep the distance: ' + str(len(idx))
        txt2 = 'Too close: ' + str(len(bbs) - len(idx))
        cv2.putText(text, txt1, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)
        cv2.putText(text, txt2, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (250, 250, 250), 1)

        if image.shape[1] < 720:
            black_i = np.zeros((720, 180, 3), np.uint8)
            concats_1 = cv2.vconcat([black_i, image])
            concats_2 = cv2.vconcat([concats_1, black_i])
        else:
            concats_2 = image

        concats = cv2.vconcat([bew_img, text])
        img = cv2.hconcat([concats_2, concats])

        image_resized = cv2.resize(img, (int(0.5 * self.img_size[0]), int(0.5 * self.img_size[1])))
        return image_resized

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

    def scaling(self, centerp, frame_width):
        # TODO move to feedback / other package
        if len(centerp) == 0:
            # TODO
            warnings.warn('CenterP length is 0, returning')
            return None, None, None
        max_vals = np.amax(centerp, axis=0)
        main_factor = 720 / (frame_width)
        mini_factor_x = 360 / (max_vals[0] + 30)
        mini_factor_y = 480 / (max_vals[1] + 30)
        return main_factor, mini_factor_x, mini_factor_y
