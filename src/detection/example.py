import os
import time

import cv2

import detection

settings = {'use_gpu': None, 'display_results': True, 'display_fps': False,
            'display_centers': True, 'display_boxes': True}


def run_test():
    # Create detector
    detector = detection.create_detector(use_gpu=settings.get('use_gpu'))

    # Print info
    print(detector.__class__.__name__)
    # Check with ONNXBackend if GPU is turned on
    from detection.utils.onnxbackend import ONNXBackend
    print(f'ONNX uses GPU: {ONNXBackend.use_gpu}')
    time.sleep(3)

    # Read images
    path = 'files/images/'
    file_names = os.listdir(path)
    images = []
    for file_name in file_names:
        bgr_image = cv2.imread(path + file_name)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        images.append(rgb_image)

    # Start timer
    _1 = None
    if settings.get('display_fps'):
        _1 = time.time_ns()
        print('Running...')

    for image in images:
        # Detect
        bounding_boxes = detector.detect(image)

        if settings.get('display_results'):
            for bounding_box in bounding_boxes:
                if settings.get('display_boxes'):
                    top_left = bounding_box.x, bounding_box.y
                    bottom_right = bounding_box.x + bounding_box.w, bounding_box.y + bounding_box.h
                    cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2)

                if settings.get('display_centers'):
                    center = bounding_box.x + bounding_box.w // 2, bounding_box.y + bounding_box.h // 2
                    cv2.circle(image, center, 6, (0, 255, 0), 8)
                    cv2.circle(image, center, 4, (255, 0, 255), 4)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('im', bgr_image)
            cv2.waitKey(10_000)
            cv2.destroyWindow('im')

    # End timer, print FPS
    if settings.get('display_fps'):
        _2 = time.time_ns()
        diff = (_2 - _1) / 1e9
        print(f'FPS: {len(images) / diff}')
    # End


if __name__ == '__main__':
    print(f'Settings: {settings}')
    assert not settings.get('display_fps') or not settings.get('display_results'), "FPS can't be calculated when " \
                                                                                   "results are displayed "
    run_test()
