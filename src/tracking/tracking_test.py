import unittest
import numpy as np
import cv2 as cv
from src.detection.boundingbox import BoundingBox
from src.tracking.tracker import BBoxFilter
from src.tracking.trackers.motpyTracker import MotpyTracker

width = 800
height = 600
dt = 1 / 15.0


def get_dummy():
    return BoundingBox(np.random.rand(1) * width,
                       np.random.rand(1) * height,
                       np.random.rand(1) / 6 * width,
                       np.random.rand(1) / 4 * height)


def draw_bounding_box(frame, bbox: BoundingBox, color=None):
    thickness = 3
    if color is None:
        color = (255, 255, 255)
        thickness = 1

    cv.rectangle(frame,
                 (int(bbox.x), int(bbox.y)),
                 (int(bbox.x + bbox.w), int(bbox.y + bbox.h)),
                 color, thickness, cv.LINE_8)


def alter_bbox(bbox: BoundingBox):
    d = dt * 5e-1
    new_bbox = BoundingBox(bbox.x + np.random.uniform(-d, d) * width,
                           bbox.y + np.random.uniform(-d, d) * width,
                           bbox.w + np.random.uniform(-d / 3, d / 3) * width,
                           bbox.h + np.random.uniform(-d / 3, d / 3) * width)
    return BoundingBox(min(max(new_bbox.x, 0), width),
                       min(max(new_bbox.y, 0), height),
                       min(max(new_bbox.w, 20), width // 2),
                       min(max(new_bbox.h, 20), height // 3))


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.dummy_bboxes = []
        for i in range(5):
            self.dummy_bboxes.append(get_dummy())

    def run_simple_test(self):
        for step in range(1000):
            frame = np.zeros((height, width, 3), np.uint8)
            if step % 50 == 0:
                self.dummy_bboxes.append(get_dummy())
            for i in range(len(self.dummy_bboxes)):
                self.dummy_bboxes[i] = alter_bbox(self.dummy_bboxes[i])
            for bbox in self.dummy_bboxes:
                draw_bounding_box(frame, bbox)
            people = self.tracker.track(self.dummy_bboxes)
            for person in people:
                draw_bounding_box(frame, person.bbox, person.color)
            cv.imshow('frame', frame)
            cv.waitKey(int(1000 * dt))

    def test_MotpyTracker(self):
        self.tracker = MotpyTracker(dt, BBoxFilter(img_size=(width, height),max_aspect=1, min_rel_height=0.1))
        self.run_simple_test()
