"""Commit 41ccf18"""
from typing import List

import cv2

from src.detection import Detector, BoundingBox
import numpy as np

from .backend import Backend

"""Based on commit 41ccf18 in
    https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3"""

# AP ~35, on cpu 30+ FPS (but uses cuda?!) on gpu 20 FPS (slower?!)
class TinyYoloV3(Detector):
    input_shape = (1, 3, 416, 416)
    onnx_file_name = 'files/yolov3/tiny-yolov3-11.onnx'

    def __init__(self):
        self.__sess = Backend.get_inference_session(self.onnx_file_name)
        self.stuff = None

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        im_data, im_size = prep(image)
        self.stuff = im_data, im_size
        out = self.__inference()
        return []

    def __inference(self) -> List[np.ndarray]:
        # copied from yolov4 - based on it
        # Step 3: Inference
        outputs = self.__sess.get_outputs()
        output_names = list(map(lambda out: out.name, outputs))
        input_1_name = self.__sess.get_inputs()[0].name
        input_2_name = self.__sess.get_inputs()[1].name

        im_data, im_size = self.stuff

        output = self.__sess.run(output_names, {input_1_name: im_data, input_2_name: im_size})
        # print("Output shape:", list(map(lambda detection: detection.shape, output)))

        # Output has:
        # 3 'heatmaps' at resolution 52x52, 26x26 and 13x13
        # Each heatmap contains 85 values for each of 3 anchors(?)
        # These 85 values are: x, y, h, w, object confidence, 80 * [class] confidence
        return output



def prep(image):
    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(image, size):
        """resize image with unchanged aspect ratio using padding"""
        iw, ih = image.shape[:2]
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh))
        new_image = cv2.resize(image, size)
        # new_image = Image.new('RGB', size, (128, 128, 128))
        # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def preprocess(img):
        model_image_size = (416, 416)
        boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    image_data = preprocess(image)
    image_size = np.array([image.shape[1], image.shape[0]], dtype=np.float32).reshape(1, 2)
    return image_data, image_size


def post():
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
