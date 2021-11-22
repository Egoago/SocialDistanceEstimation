from typing import Tuple, List

import cv2
import numpy as np

from tracking import Person


class SocialDistanceEstimator:
    def __init__(self, dt: float, img_size: Tuple[int, int]):
        from detection import create_detector
        self.detector = create_detector()

        import tracking as tr
        self.img_size = img_size
        bbox_filter = tr.BBoxFilter(img_size=img_size,
                                    max_aspect=0.8,
                                    min_rel_height=0.1)
        self.tracker = tr.create_tracker(dt, bbox_filter=bbox_filter)

        from projection import create_calibrator, Intrinsics
        self.calibrator = create_calibrator(Intrinsics(res=np.array(img_size)), method='least_squares')

        self.settings = {
            # Draw a high-contrast disk at the center of the person's box
            'display_centers': False,
            # Draw a bounding box around the person
            'display_bounding_boxes': True,
            # Draw a circle in 3D around at each person's feet
            'display_proximity': True,
            # Amount of bounding boxes to gather before calibration
            'calibrate_at_bounding_boxes_count': 2000,
        }
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
        for person in people:
            if self.settings.get('display_bounding_boxes'):
                top_left, bottom_right = person.bbox.corners()
                cv2.rectangle(image, top_left, bottom_right, person.color, 2)

            if self.camera is not None and self.settings.get('display_proximity'):
                import projection as proj
                res = 20
                radius = 1000
                center = proj.back_project(np.array(proj.opencv2opengl(person.bbox.bottom(), self.img_size[1])),
                                           self.camera)
                pixels = []
                for i in np.linspace(0, 2 * np.pi.real, res):
                    point = center + np.array([np.cos(i), 0, np.sin(i)], dtype=float) * radius
                    pixel = proj.opengl2opencv(tuple(proj.project(point, self.camera)[0]), self.img_size[1])
                    pixels.append(pixel)
                cv2.polylines(image, np.int32([pixels]), True, (255, 128, 0), 2)

            if self.settings.get('display_centers'):
                center = person.bbox.x + person.bbox.w // 2, person.bbox.y + person.bbox.h // 2
                cv2.circle(image, center, 6, (0, 255, 0), 8)
                cv2.circle(image, center, 4, (255, 0, 255), 4)

        # CV2 can display BGR images
        image_resized = cv2.resize(image, (960, 540))
        return image_resized

    def __calibrate(self, people: List[Person]):
        import projection as proj
        for person in people:
            self.p_top[self.bounding_boxes_count] = proj.opencv2opengl(person.bbox.top(), self.img_size[1])
            self.p_bottom[self.bounding_boxes_count] = proj.opencv2opengl(person.bbox.bottom(), self.img_size[1])
            self.bounding_boxes_count += 1
            if self.bounding_boxes_count == self.settings['calibrate_at_bounding_boxes_count']:
                # from src.projection.calibrators.test.drawing import draw_2d_points
                # ax = draw_2d_points(self.p_bottom, c='darkgreen', last=False)
                # draw_2d_points(self.p_top, c='darkred', ax=ax, res=self.img_size)
                self.camera = self.calibrator.calibrate(p_top=self.p_top, p_bottom=self.p_bottom)
                return
