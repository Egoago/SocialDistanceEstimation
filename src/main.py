import logging
from typing import Tuple

import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)-6s %(message)s",
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class SocialDistanceEstimator:
    def __init__(self, dt: float, img_size: Tuple[int, int]):
        from detection import create_detector
        self.detector = create_detector()

        import tracking as tr
        self.img_size = img_size
        bbox_filter = tr.BBoxFilter(img_size=img_size,
                                    max_aspect=0.8,
                                    min_rel_height=0.1)
        self.tracker = tr.create_tracker(dt, bbox_filter=bbox_filter)

        from projection import create_calibrator, Intrinsics
        self.calibrator = create_calibrator(Intrinsics(res=np.array(img_size)), method='least_squares')

        self.settings = {'display_centers': True,
                         'display_boxes': True,
                         'display_proximity': True,
                         'calibrate_at_bounding_boxes_count': 2000}
        self.p_bottom = np.zeros((self.settings['calibrate_at_bounding_boxes_count'], 2), dtype=float)
        self.p_top = np.zeros((self.settings['calibrate_at_bounding_boxes_count'], 2), dtype=float)
        self.bounding_boxes_count = 0
        self.camera = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        bounding_boxes = self.detector.detect(image)
        people = self.tracker.track(bounding_boxes)

        if self.bounding_boxes_count < self.settings['calibrate_at_bounding_boxes_count']:
            self.__calibrate(people)

        im = self.__create_image(image=image, people=people)
        return im

    def __create_image(self, image, people):
        for person in people:
            if self.settings.get('display_boxes'):
                top_left, bottom_right = person.bbox.corners()
                cv2.rectangle(image, top_left, bottom_right, person.color, 2)

                if self.camera is not None and self.settings.get('display_proximity'):
                    import projection as proj
                    res = 20
                    radius = 1000
                    center = proj.back_project(np.array(proj.opencv2opengl(person.bbox.bottom(), self.img_size[1])),
                                               self.camera)
                    pixels = []
                    for i in np.linspace(0, 2 * np.pi.real, res):
                        point = center + np.array([np.cos(i), 0, np.sin(i)], dtype=float) * radius
                        pixel = proj.opengl2opencv(tuple(proj.project(point, self.camera)[0]), self.img_size[1])
                        pixels.append(pixel)
                    cv2.polylines(image, np.int32([pixels]), True, (255, 128, 0), 2)

            if self.settings.get('display_centers'):
                center = person.bbox.x + person.bbox.w // 2, person.bbox.y + person.bbox.h // 2
                cv2.circle(image, center, 6, (0, 255, 0), 8)
                cv2.circle(image, center, 4, (255, 0, 255), 4)

        # CV2 can display BGR images
        image_resized = cv2.resize(image, (960, 540))
        return image_resized

    def __calibrate(self, people):
        import projection as proj
        for person in people:
            self.p_top[self.bounding_boxes_count] = proj.opencv2opengl(person.bbox.top(), self.img_size[1])
            self.p_bottom[self.bounding_boxes_count] = proj.opencv2opengl(person.bbox.bottom(), self.img_size[1])
            self.bounding_boxes_count += 1
            if self.bounding_boxes_count == self.settings['calibrate_at_bounding_boxes_count']:
                # from src.projection.calibrators.test.drawing import draw_2d_points
                # ax = draw_2d_points(self.p_bottom, c='darkgreen', last=False)
                # draw_2d_points(self.p_top, c='darkred', ax=ax, res=self.img_size)
                self.camera = self.calibrator.calibrate(p_top=self.p_top, p_bottom=self.p_bottom)
                return


def main():
    frames_to_process = 50  # Should be no more than 4_000
    video_path = 'files/videos/OxfordTownCentreDataset.avi'
    output_video_path = 'files/output.avi'
    # "Oxford Town Centre Dataset" video - Source:
    # https://drive.google.com/file/d/1UMIcffhxGw1aCAyztNWlslHHtayw9Fys/view
    # from https://github.com/DrMahdiRezaei/DeepSOCIAL
    # Dataset:
    # """B. Benfold and I. Reid, "Stable multi-target tracking in real-time surveillance video,"
    # CVPR 2011, 2011, pp. 3457-3464, doi: 10.1109/CVPR.2011.5995667."""

    logger.debug('Startup')
    logger.info(f'Loading video {video_path}')
    video = cv2.VideoCapture(video_path)
    major_ver = cv2.__version__.split('.')[0]  # noqa  # __version__ works, stopping
    assert float(major_ver) >= 3, 'video.get(cv2.CAP_PROP_FPS) needs opencv major version >= 3.'
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dt = 1000 / fps
    logger.debug(f'fps {fps} width {width} height {height}')
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
    video.release()
    logger.info(f'{len(frames)} frames read successfully')

    logger.info('Starting SocialDistanceEstimation')
    sde = SocialDistanceEstimator(dt=dt, img_size=(width, height))
    logger.info('SocialDistanceEstimation setup complete')
    processed_frames = []
    for frame in frames:
        im = sde(frame)
        processed_frames.append(im)
        if len(processed_frames) % 1_000 == 0:
            logger.debug(f'{len(processed_frames)} frames processed...')
    logger.info('SocialDistanceEstimation processing complete')

    del frames  # Free up memory
    # CV2 expects HxWxC
    out = cv2.VideoWriter(filename=output_video_path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=fps,
                          frameSize=(int(0.5*height), int(0.5*width)), isColor=True)
    for image in processed_frames:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(bgr_image)
        cv2.imshow('SocialDistanceEstimation', bgr_image)
        cv2.waitKey(int(dt))
    out.release()
    cv2.destroyAllWindows()
    logger.debug('Exit')


if __name__ == '__main__':
    main()
