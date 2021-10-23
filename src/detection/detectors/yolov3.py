import warnings
from typing import List, Tuple

import numpy as np

from src.detection import Detector, BoundingBox
from .backend import Backend

from .preprocessor import Preprocessor, OperationInfo


# TODO finish if needed
class TinyYoloV3(Detector):
    """
    Based on commit 41ccf18 in
        "https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3"
    Has AP ~35, 30+ FPS CPU (but sometimes uses cuda?!), 20 FPS GPU (slower?!)
    Onnx source:
        "https://github.com/onnx/models/blob/master/vision/
            object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"
    """

    input_shape = (1, 3, 416, 416)  # Attention! TinyYolo expects B x C x H or W x H or W
    onnx_file_name = 'files/yolov3/tiny-yolov3-11.onnx'

    def __init__(self):
        warnings.warn(f'{self.__class__.__name__} is still under development')
        self.__sess = Backend.get_inference_session(self.onnx_file_name)

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        image = image.copy()

        image, info = self.__preprocess(image)
        outputs = self.__inference(image, image.shape)
        bounding_boxes = self.__postprocess(outputs, info)

        return bounding_boxes

    def __inference(self, image, shape) -> List[np.ndarray]:
        outputs = self.__sess.get_outputs()
        output_names = list(map(lambda out: out.name, outputs))
        input_1_name = self.__sess.get_inputs()[0].name
        input_2_name = self.__sess.get_inputs()[1].name

        output = self.__sess.run(output_names, {input_1_name: image, input_2_name: shape})
        # print("Output shape:", list(map(lambda detection: detection.shape, output)))

        # Output has:
        # ???
        # """
        # The model has 3 outputs.
        #   boxes: (1x'n_candidates'x4), the coordinates of all anchor boxes,
        #   scores: (1x80x'n_candidates'), the scores of all anchor boxes per class,
        #   indices: ('nbox'x3), selected indices from the boxes tensor.
        #       The selected index format is (batch_index, class_index, box_index).
        # """
        return output

    @classmethod
    def __postprocess(cls, outputs, info: OperationInfo) -> List[BoundingBox]:
        # TODO implement correctly
        warnings.warn('TinyYoloV3 postprocess is probably not implemented correctly')
        boxes, scores, indices = outputs
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices[0]:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        return []

    @classmethod
    def __preprocess(cls, image) -> Tuple[np.ndarray, OperationInfo]:
        # TODO check and fix
        warnings.warn("TinyYoloV3 has a weird input shape which does not conform to preprocess_image's expected params")
        image, info = Preprocessor.preprocess_image(image, cls.input_shape)
        return image, info
