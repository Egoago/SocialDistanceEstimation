from typing import List, Tuple

import numpy as np

from ..detector import Detector
from ..boundingbox import BoundingBox
from ..utils import ONNXBackend, Preprocessor, OperationInfo, Coco


class TinyYoloV3(Detector):
    """
    Based on commit 41ccf18 in
        "https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3"
    Has AP ~35, 30+ FPS CPU (but sometimes uses cuda?!), 20 FPS GPU (slower?!)
    Onnx source:
        "https://github.com/onnx/models/blob/master/vision/
            object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"
    """

    input_shape = (1, 3, 416, 416)  # Attention! TinyYolo expects Batch x Channel x Height x Width
    onnx_file_name = 'files/yolov3/tiny-yolov3-11.onnx'

    def __init__(self, use_gpu=None):
        super().__init__(use_gpu)

        ONNXBackend.use_gpu = use_gpu
        self.__sess = ONNXBackend.get_inference_session(self.onnx_file_name)

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        image = image.copy()

        image, info = self.__preprocess(image)
        outputs = self.__inference(image, image.shape)
        bounding_boxes = self.__postprocess(outputs, info)

        return bounding_boxes

    def __inference(self, image, shape) -> List[np.ndarray]:
        outputs = self.__sess.get_outputs()
        output_names = list(map(lambda out: out.name, outputs))
        input_0_name = self.__sess.get_inputs()[0].name
        input_1_name = self.__sess.get_inputs()[1].name

        # H x W x C -> C x H x W
        image = image.transpose((2, 0, 1))
        image = image.reshape((1, *image.shape))
        image_shape = np.array([shape[1], shape[0]], dtype=np.float32).reshape(1, 2)

        output = self.__sess.run(output_names, input_feed={input_0_name: image, input_1_name: image_shape})
        # print("Output shape:", list(map(lambda detection: detection.shape, output)))

        # Output has:
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
        scale, pad_each_x, pad_each_y = info

        boxes, scores, indices = outputs
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices[0]:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        info = zip(out_boxes, out_scores, out_classes)
        bounding_boxes = []
        for _box, _score, _cls in info:
            # _box -> y1, x1, y2, x2
            # _score -> float
            # _cls -> int (COCO)

            if Coco.class_names.get(_cls) != 'person':
                # print(Coco.class_names.get(_cls))
                continue
            if _score < 0.1:
                continue
            y1, x1, y2, x2 = _box
            y1 = (y1 - pad_each_y) / scale
            y2 = (y2 - pad_each_y) / scale
            x1 = (x1 - pad_each_x) / scale
            x2 = (x2 - pad_each_x) / scale
            w = x2-x1
            h = y2-y1
            bounding_boxes.append(BoundingBox(x=int(x1), y=int(y1), w=int(w), h=int(h)))
        return bounding_boxes

    @classmethod
    def __preprocess(cls, image) -> Tuple[np.ndarray, OperationInfo]:
        image, info = Preprocessor.preprocess_image(image,
                                                    (cls.input_shape[0], cls.input_shape[2],
                                                     cls.input_shape[3], cls.input_shape[1]))
        return image, info
