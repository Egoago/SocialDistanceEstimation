# SocialDistanceEstimation

Detects people on the input video, calculates their distances
and gives feedback about their social distancing.

## Installation

### Requirements

- Windows 10
- Python 3.8

### Steps

1. Clone repository

   For example with Git CLI: `git clone https://github.com/Egoago/SocialDistanceEstimation`
2. Get dependencies

   ***Recommended:***
    - Get example input videos - for example the [Oxford Town Centre Dataset video from https://github.com/DrMahdiRezaei/DeepSOCIAL](https://drive.google.com/file/d/1UMIcffhxGw1aCAyztNWlslHHtayw9Fys/view)
    - Get [YOLOv4 onnx file](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx)

   ***Optional:***
    - If using GPU: CUDA drivers for using Nvidia GPUs to improve runtime, then install onnxruntime-gpu==1.7.0 instead of the regular onnxruntime in Step 3
    - If using an IDE: Change working directory from `SocialDistanceEstimation/src` to `SocialDistanceEstimation`
3. Install Python dependencies with

   Recommended: in a new Conda virtual environment with Python 3.8

   For example with Pip3: `pip install -r files/requirements.txt`

## Usage

`python -m src.main`

Run `python -m src.main --help` from `SocialDistanceEstimation` to see command line arguments.
