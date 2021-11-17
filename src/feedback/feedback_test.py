import feedback as f
from src.detection.boundingbox import BoundingBox
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == '__main__':
    bb1 = BoundingBox(125, 116, 50, 20)
    bb2 = BoundingBox(122, 170, 50, 20)
    bb3 = BoundingBox(150, 144, 50, 20)
    bb4 = BoundingBox(50, 550, 50, 20)
    bb5 = BoundingBox(540, 317, 50, 20)
    bb6 = BoundingBox(317, 272, 50, 20)
    bb7 = BoundingBox(443, 111, 50, 20)

    bbs = (bb1, bb2, bb3, bb4, bb5, bb6, bb7)
    centerp = np.array([[125, 116], [122, 170], [150, 144], [50, 550], [540, 317], [317, 272], [443, 111]])

    path = 'src/feedback/output_frames'

    frames = []
    f.feedback_image(None, bbs, centerp, 150, 1, 640, path)
    f.feedback_image(None, bbs, centerp, 150, 2, 640, path)

    for i in range(20):
        files = os.listdir(path)
        for file in files:
            name = path + '/' + file
            frames.append(cv2.imread(name))

    f.video_maker('src/feedback/Test_video.mp4', frames, 10)






