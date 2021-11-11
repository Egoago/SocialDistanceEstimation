import logging

import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)-6s %(message)s",
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class SocialDistanceEstimator:
    def __init__(self):
        from detection import create_detector
        self.detector = create_detector()

        self.settings = {'display_results': True, 'display_centers': True, 'display_boxes': True}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        bounding_boxes = self.detector.detect(image)
        if self.settings.get('display_results'):
            for bounding_box in bounding_boxes:
                if self.settings.get('display_boxes'):
                    top_left = bounding_box.x, bounding_box.y
                    bottom_right = bounding_box.x + bounding_box.w, bounding_box.y + bounding_box.h
                    cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2)

                if self.settings.get('display_centers'):
                    center = bounding_box.x + bounding_box.w // 2, bounding_box.y + bounding_box.h // 2
                    cv2.circle(image, center, 6, (0, 255, 0), 8)
                    cv2.circle(image, center, 4, (255, 0, 255), 4)

            # CV2 can display BGR images
            image_resized = cv2.resize(image, (960, 540))
            return image_resized


def main():
    frames_to_process = 2_000  # Should be no more than 4_000
    video_path = 'files/videos/OxfordTownCentreDataset.avi'
    # "Oxford Town Centre Dataset" video - Source:
    # https://drive.google.com/file/d/1UMIcffhxGw1aCAyztNWlslHHtayw9Fys/view
    # from https://github.com/DrMahdiRezaei/DeepSOCIAL
    # Dataset:
    # """B. Benfold and I. Reid, "Stable multi-target tracking in real-time surveillance video,"
    # CVPR 2011, 2011, pp. 3457-3464, doi: 10.1109/CVPR.2011.5995667."""

    logger.debug('Startup')
    logger.info(f'Loading video {video_path}')
    video = cv2.VideoCapture(video_path)
    fps = 25  # based on file properties
    logger.debug('Video opened successfully')

    success = True
    frames = []
    while success:
        success, image = video.read()
        if not success:
            raise ValueError(f'Error reading frame {len(frames) + 1}')
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(frames) % 1_000 == 0:
            logger.debug(f'{len(frames)} frames read...')
        if len(frames) >= frames_to_process:
            # Python can't seem to hold more than ~4000 frames in a list
            # loading and storing new frames slows down exponentially
            break
    logger.info(f'{len(frames)} frames read successfully')

    logger.info('Starting SocialDistanceEstimation')
    sde = SocialDistanceEstimator()
    logger.info('SocialDistanceEstimation setup complete')
    processed_frames = []
    for frame in frames:
        im = sde(frame)
        processed_frames.append(im)
        if len(processed_frames) % 1_000 == 0:
            logger.debug(f'{len(processed_frames)} frames processed...')
    logger.info('SocialDistanceEstimation processing complete')

    del frames  # Free up memory
    dt = 1000 / fps
    for image in processed_frames:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('SocialDistanceEstimation', bgr_image)
        cv2.waitKey(int(dt))
    logger.debug('Exit')


if __name__ == '__main__':
    main()
