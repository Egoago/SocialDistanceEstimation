# Documentation of Detection package

To use this package, `import detection` after **downloading the [dependencies](#Dependencies)**.

`detection` contains `Detector`, `BoundingBox`, `create_detector`.

## Overview

`Detector` is the interface of detectors.
See it's description in [`detector.py`](detector.py).

`BoundingBox` represents a 2D bounding box of an object.
See it's description in [`boundingbox.py`](boundingbox.py).

Call `create_detector` to get a detector. This has an optional argument, `use_gpu`.
**GPU acceleration can have additional dependencies**.
See `create_detector`'s description in [`__init__.py`](__init__.py).

## Example code

```python
import cv2
import detection

# Load image
bgr_image = cv2.imread('image.png')  # CV2 reads images as BGR

# Get a detector
detector = detection.create_detector()  # Optional parameter use_gpu not set

# Convert image to RGB because detect() expects an RGB image
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

bounding_boxes = detector.detect(rgb_image)
# bounding_boxes has type List[BoundingBox]
```

or see [`example.py`](example.py).

## Dependencies

To use any detector, some files must be downloaded.

The current detector, `YoloV4` requires three files:

| File name             | Directory         | Version used                              | Download link                                                                                                                     |
|-----------------------|-------------------|-------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------|
| `yolov4.onnx`         | `files/yolov4`    | fd818b3b17ae7affb7c4a4f136f69401e883e296  | [link](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx)                  |
|`yolov4_anchors.txt`   | `files/yolov4`    | c15d7731c993852a52ea3453bccdb4288d692551  | [link](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/dependencies/yolov4_anchors.txt)    |
|`coco.names`           | `files/coco`      | f1cb2cf21b226d830932f5db7a1379e3f92ba4dd  | [link](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/dependencies/coco.names)            |
