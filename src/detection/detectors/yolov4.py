import warnings
from typing import List, NamedTuple, Tuple
import cv2
import numpy as np
from scipy import special
import colorsys
import random

from .backend import Backend
from .preprocessor import OperationInfo, Preprocessor
from ..boundingbox import BoundingBox
from ..detector import Detector

# Probability = objectness confidence * class confidence
Bbox = NamedTuple('Bbox', x1=float, y1=float, x2=float, y2=float, probability=float, category=int)


def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class Coco:
    class_names = read_class_names('files/coco/coco.names')


def rescale_boxes(boxes: List[Bbox], info: OperationInfo):
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * info.scale + info.diff_width
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * info.scale + info.diff_height
    return boxes


# Convert and return
def Bbox2BoundingBox(bbox: Bbox) -> BoundingBox:
    x1, y1, x2, y2 = [int(x) for x in bbox[:-2]]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return BoundingBox(x=x, y=y, w=w, h=h)


def is_person(bbox: Bbox) -> bool:
    if bbox.probability > 0.5 and Coco.class_names.get(int(bbox.category)) == 'person':
        return True
    else:
        return False


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
    strides = np.array([8, 16, 32])
    x_y_scale = [1.2, 1.1, 1.05]  #

    anchors = get_anchors('files/yolov4/yolov4_anchors.txt')
    onnx_file_name = 'files/yolov4/yolov4.onnx'

    def __init__(self):
        self.__sess = Backend.get_inference_session(self.onnx_file_name)

    def detect(self, image: np.array) -> List[BoundingBox]:
        image = image.copy()

        image, info = self.__preprocess(image)
        outputs = self.__inference(image)
        bounding_boxes = self.__postprocess(outputs, info, image)  # TODO remove image

        return bounding_boxes

    @classmethod
    def __preprocess(cls, image) -> Tuple[np.ndarray, OperationInfo]:
        image, info = Preprocessor.preprocess_image(image, YoloV4.input_shape)
        return image, info

    def __inference(self, image) -> List[np.ndarray]:
        # Step 3: Inference
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
        # These 85 values are: center_x, center_y, h, w, object confidence, 80 * [class] confidence
        return outputs

    @classmethod
    def __postprocess(cls, outputs, info: OperationInfo, image) -> List[BoundingBox]:
        detections = PostProcessor.generate_detections(outputs, YoloV4.anchors, YoloV4.strides, YoloV4.x_y_scale)
        bboxes = PostProcessor.generate_and_adjust_bboxes(detections, 0.25, info)
        bboxes = PostProcessor.nms(bboxes, 0.213, method='nms')

        # Show outputs
        image = cls.draw_bbox(image, bboxes)
        cv2.imshow('img', image)
        cv2.waitKey(10_000)
        cv2.destroyWindow('img')

        # Convert outputs
        bounding_boxes = []
        for bbox in bboxes:
            b = Bbox(*bbox)
            if is_person(b):
                bounding_boxes.append(Bbox2BoundingBox(b))
        return bounding_boxes

    @staticmethod
    def draw_bbox(image, bboxes, show_label=True):
        """
            bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
            """

        image = image.copy()
        num_classes = len(Coco.class_names)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        font_scale = 0.5

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coords = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coords[0], coords[1]), (coords[2], coords[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (Coco.class_names[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, font_scale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        return image


class PostProcessor:
    @staticmethod
    def generate_detections(output, anchors, strides, x_y_scale) -> np.ndarray:
        """Generates an array of every detection with shape (-1, 85)"""
        for i, heatmap in enumerate(output):
            heatmap_side_length = heatmap.shape[1]  # assert heatmap.shape[1] == heatmap.shape[2]

            # heatmaps contain: Batch x H x W x Anchor x Value
            # Values are: center_x, center_y, h, w, object confidence, 80 x class confidence
            heatmap_of_x_y = heatmap[:, :, :, :, 0:2]
            heatmap_of_h_w = heatmap[:, :, :, :, 2:4]

            xy_grid = np.meshgrid(np.arange(heatmap_side_length), np.arange(heatmap_side_length))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)
            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float)
            # A grid shaped like output, except -1 is 2
            # A matrix with sides heatmap_side_length, containing triples of two numbers
            # going from (0 0)(0 0)(0 0) to (51 51)(51 51)(51 51)

            predictions_x_y = ((special.expit(heatmap_of_x_y) * x_y_scale[i]) - 0.5 *
                               (x_y_scale[i] - 1) + xy_grid) * strides[i]
            predictions_h_w = (np.exp(heatmap_of_h_w) * anchors[i])

            # Put x, y, h, w back into heatmap
            heatmap[:, :, :, :, 0:4] = np.concatenate([predictions_x_y, predictions_h_w], axis=-1)

        # detections is output, but
        # each heatmap is reshaped into a list of values with length of 85
        # These 85 values are: center_x, center_y, h, w, object confidence, 80 * [class] confidence
        detections = [np.reshape(heatmap, (-1, heatmap.shape[-1])) for heatmap in output]
        detections = np.concatenate(detections, axis=0)
        # detections is the array of every detection with shape -1, 85
        return detections

    @staticmethod
    def generate_and_adjust_bboxes(detections, score_threshold, info: OperationInfo):
        """rework boxes, work with confidence, remove boundary boxes with a low detection probability"""
        valid_scale = [0, np.inf]

        predicted_x_y_h_ws = detections[:, 0:4]  # (center_x, center_y, h, w)'s
        predicted_objectness_ = detections[:, 4]  # objectness'
        predicted_class_confidences = detections[:, 5:]

        # Swap height and width
        predicted_x_y_w_hs = predicted_x_y_h_ws.copy()
        predicted_x_y_w_hs[:, [2, 3]] = predicted_x_y_w_hs[:, [3, 2]]

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
        # TODO is padding correct?

        # clip some boxes that are out of range
        # TODO use input shape
        coords = np.concatenate([np.maximum(predicted_coords_transformed[:, :2], [0, 0]),
                                 np.minimum(predicted_coords_transformed[:, 2:], [416 - 1, 416 - 1])], axis=-1)
        invalid_mask = np.logical_or((coords[:, 0] > coords[:, 2]),
                                     (coords[:, 1] > coords[:, 3]))
        if np.sum(invalid_mask) > 0:
            warnings.warn(f'{np.sum(invalid_mask) / len(invalid_mask) * 100}% of generated detections are outside')
        coords[invalid_mask] = 0

        # Create a mask for boxes with invalid scales
        scale_of_bboxes = np.sqrt(
            np.multiply.reduce(coords[:, 2:4] - coords[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < scale_of_bboxes), (scale_of_bboxes < valid_scale[1]))
        if np.sum(scale_mask) < len(scale_mask):
            warnings.warn(f'{(len(scale_mask) - np.sum(scale_mask)) / len(scale_mask) * 100}% of generated detections have invalid scale')

        # Create a mask for boxes with low scores
        classes = np.argmax(predicted_class_confidences, axis=-1)
        scores = predicted_objectness_ * predicted_class_confidences[np.arange(len(coords)), classes]
        score_mask = scores > score_threshold

        # Remove invalid boxes
        mask = np.logical_and(scale_mask, score_mask)
        coords, scores, classes = coords[mask], scores[mask], classes[mask]

        bboxes = np.concatenate([coords, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return bboxes

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
    def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
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
                iou = PostProcessor.calculate_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
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
