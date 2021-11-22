from typing import List, Tuple
import numpy as np
from scipy import special

from ..detector import Detector
from ..boundingbox import BoundingBox
from ..utils import ONNXBackend, Preprocessor, OperationInfo, Coco


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape((3, 3, 2))


# on cpu 1-3 FPS, on gpu 8-12 FPS
class YoloV4(Detector):
    """
    YoloV4 detector in ONNX. Achievable results are mAP50 of 52.32 on the COCO 2017 dataset and 41.7 FPS on a Tesla 100.

    Original source of code: Commit 857a343
        "https://github.com/onnx/models/blob/master/vision/
            object_detection_segmentation/yolov4/dependencies/inference.ipynb"
    Dependencies at:
        "https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4/dependencies"
    Onnx source:
        "https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx"
    """

    input_shape = (1, 416, 416, 3)  # (batch_size, height, width, channels)
    strides = (8, 16, 32)
    x_y_scale = (1.2, 1.1, 1.05)  # ?
    anchors = get_anchors('files/yolov4/yolov4_anchors.txt')
    onnx_file_name = 'files/yolov4/yolov4.onnx'

    # Parameters
    score_threshold = 0.1  # Default: 0.25
    iou_threshold = 0.3  # Default: 0.213
    valid_scale = (16, 208)  # Default: (0, 416 // 2)

    def __init__(self, use_gpu=None):
        super().__init__(use_gpu)

        ONNXBackend.use_gpu = use_gpu
        self.__sess = ONNXBackend.get_inference_session(self.onnx_file_name)

    def detect(self, image: np.array) -> List[BoundingBox]:
        image = image.copy()

        image, info = self.__preprocess(image)
        outputs = self.__inference(image)
        bounding_boxes = self.__postprocess(outputs, info)

        return bounding_boxes

    @classmethod
    def __preprocess(cls, image) -> Tuple[np.ndarray, OperationInfo]:
        image, info = Preprocessor.preprocess_image(image, cls.input_shape)
        return image, info

    def __inference(self, image) -> List[np.ndarray]:
        output_metadata = self.__sess.get_outputs()
        output_names = list(map(lambda out: out.name, output_metadata))
        input_name = self.__sess.get_inputs()[0].name

        image = image.reshape(self.input_shape)
        assert image.dtype == np.float32, f'{image.dtype} != {np.float32}'

        outputs = self.__sess.run(output_names, {input_name: image})
        # print("Output shape:", list(map(lambda detection: detection.shape, output)))

        # Output has:
        # 3 'heatmaps' at resolution 52x52, 26x26 and 13x13
        # Each heatmap contains 85 values for each of 3 anchors(?)
        # These 85 values are: center_x, center_y, w, h, object confidence, 80 * [class] confidence
        # Documentation says "(x, y, h, w [, ...])", but it's actually (center_x, center_y, w, h, ...)
        return outputs

    @classmethod
    def __postprocess(cls, outputs, info: OperationInfo) -> List[BoundingBox]:
        detections = YoloV4PostProcessor.generate_detections(outputs, cls.anchors, cls.strides, cls.x_y_scale)
        bboxes = YoloV4PostProcessor.generate_and_adjust_bboxes(detections, cls.score_threshold, cls.valid_scale,
                                                                info, cls.input_shape)
        bboxes = YoloV4PostProcessor.nms(bboxes, iou_threshold=cls.iou_threshold, method='nms')

        # Convert outputs
        class Bbox:
            def __init__(self, x1: float, y1: float, x2: float, y2: float, probability: float, category: int):
                self.x1 = x1
                self.y1 = y1
                self.x2 = x2
                self.y2 = y2
                self.probability = probability  # Probability = objectness confidence * class confidence
                self.category = int(category)

            def to_BoundingBox(self) -> BoundingBox:
                x, y, w, h = self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1
                return BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h))

            def is_person(self) -> bool:
                if Coco.class_names.get(int(self.category)) == 'person':
                    return True
                else:
                    return False

        bounding_boxes = []
        for bbox in bboxes:
            b = Bbox(*bbox)
            if b.is_person():
                bounding_boxes.append(b.to_BoundingBox())
        return bounding_boxes


class YoloV4PostProcessor:
    @staticmethod
    def generate_detections(output, anchors, strides, x_y_scale) -> np.ndarray:
        """Generates an array of every detection with shape (-1, 85)"""
        for i, heatmap in enumerate(output):
            heatmap_side_length = heatmap.shape[1]  # assert heatmap.shape[1] == heatmap.shape[2]

            # heatmaps contain: Batch x H? x W? x Anchor x Value - or B x W x H x A x V
            # Values are: center_x, center_y, h, w, object confidence, 80 x class confidence
            heatmap_of_x_y = heatmap[:, :, :, :, 0:2]
            heatmap_of_w_h = heatmap[:, :, :, :, 2:4]

            xy_grid = np.meshgrid(np.arange(heatmap_side_length), np.arange(heatmap_side_length))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)
            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float)
            # A grid shaped like output, except -1 is 2
            # A matrix with sides heatmap_side_length, containing triples of two numbers
            # going from (0 0)(0 0)(0 0) to (51 51)(51 51)(51 51)

            predictions_x_y = ((special.expit(heatmap_of_x_y) * x_y_scale[i]) - 0.5 *
                               (x_y_scale[i] - 1) + xy_grid) * strides[i]
            predictions_w_h = (np.exp(heatmap_of_w_h) * anchors[i])

            # Put x, y, h, w back into heatmap
            heatmap[:, :, :, :, 0:4] = np.concatenate([predictions_x_y, predictions_w_h], axis=-1)

        # detections is output, but
        # each heatmap is reshaped into a list of values with length of 85
        # These 85 values are: center_x, center_y, w, h, object confidence, 80 * [class] confidence
        detections = [np.reshape(heatmap, (-1, heatmap.shape[-1])) for heatmap in output]
        detections = np.concatenate(detections, axis=0)
        # detections is the array of every detection with shape -1, 85
        return detections

    @staticmethod
    def generate_and_adjust_bboxes(detections: np.ndarray, score_threshold: float, valid_scale: Tuple[float, float],
                                   info: OperationInfo, input_shape) -> np.ndarray:
        """rework boxes, work with confidence, remove boundary boxes with a low detection probability"""

        predicted_x_y_w_hs = detections[:, 0:4]  # (center_x, center_y, w, h)'s
        predicted_objectness_ = detections[:, 4]  # objectness'
        predicted_class_confidences = detections[:, 5:]

        # Swap height and width - documentation is wrong, output is x, y, w, h
        # predicted_x_y_w_hs = predicted_x_y_h_ws.copy()
        # warnings.warn('Not swapping')
        # predicted_x_y_w_hs[:, [2, 3]] = predicted_x_y_w_hs[:, [3, 2]]

        # (center_x, center_y, w, h) --> (x_min, y_min, x_max, y_max)
        predicted_x_y_s = predicted_x_y_w_hs[:, :2]
        predicted_w_h_s = predicted_x_y_w_hs[:, 2:]
        predicted_coords = np.concatenate([predicted_x_y_s - 0.5 * predicted_w_h_s,
                                           predicted_x_y_s + 0.5 * predicted_w_h_s], axis=-1)

        # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        scale, pad_each_x, pad_each_y = info
        predicted_coords_transformed = predicted_coords
        # Update x coordinates
        predicted_coords_transformed[:, 0::2] = (predicted_coords[:, 0::2] - pad_each_x) / scale
        # Update y coordinates
        predicted_coords_transformed[:, 1::2] = (predicted_coords[:, 1::2] - pad_each_y) / scale

        def clip(predicted_coords_transformed, info, input_shape):
            # clip some boxes that are out of range
            # Note: maybe 1 and 2 should be flipped
            orig_x_size = (input_shape[1] - 2 * info.pad_each_x) / scale - 1
            orig_y_size = (input_shape[2] - 2 * info.pad_each_y) / scale - 1
            coords = np.concatenate([np.maximum(predicted_coords_transformed[:, :2], [0, 0]),
                                     np.minimum(predicted_coords_transformed[:, 2:],
                                                [orig_x_size, orig_y_size])], axis=-1)
            invalid_mask = np.logical_or((coords[:, 0] > coords[:, 2]),
                                         (coords[:, 1] > coords[:, 3]))
            coords[invalid_mask] = 0

            # Create a mask for boxes with invalid scales
            scale_of_bboxes = np.sqrt(
                np.multiply.reduce(coords[:, 2:4] - coords[:, 0:2], axis=-1))
            scale_mask = np.logical_and((valid_scale[0] < scale_of_bboxes), (scale_of_bboxes < valid_scale[1]))

            # Create a mask for boxes with low scores
            classes = np.argmax(predicted_class_confidences, axis=-1)
            scores = predicted_objectness_ * predicted_class_confidences[np.arange(len(coords)), classes]
            score_mask = scores > score_threshold

            # Remove invalid boxes
            score_and_scale_mask = np.logical_and(scale_mask, score_mask)
            return coords[score_and_scale_mask], scores[score_and_scale_mask], classes[score_and_scale_mask]

        coords, scores, classes = clip(predicted_coords_transformed, info, input_shape)

        data = np.concatenate([coords, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return data

    @staticmethod
    def calculate_iou(box1, box2):
        """calculate the Intersection Over Union value"""
        box1 = np.array(box1)
        box2 = np.array(box2)

        boxes1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        boxes2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        left_up = np.maximum(box1[..., :2], box2[..., :2])
        right_down = np.minimum(box1[..., 2:], box2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return iou

    @staticmethod
    def nms(bboxes: np.ndarray, iou_threshold: float, sigma=0.3, method='nms'):
        """
        :param method:
        :param sigma:
        :param iou_threshold:
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        returns less bboxes

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = YoloV4PostProcessor.calculate_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes
