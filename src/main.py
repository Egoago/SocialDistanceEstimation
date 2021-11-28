import json
import logging
import os.path
from tqdm import tqdm

import cv2

from src.SocialDistanceEstimator import SocialDistanceEstimator

logging.basicConfig(format="%(asctime)s %(levelname)-6s %(message)s",
                    level=logging.ERROR,
                    datefmt='%Y-%m-%d %H:%M:%S')
project_logger = logging.getLogger('src')
logger = logging.getLogger('src.main')


def main(args):

    project_logger.setLevel(args.logging_level)

    logger.debug('Startup')
    logger.info(f'Arguments: {json.dumps(vars(args), indent=2)}')

    logger.info(f'Loading video {args.video_path}')
    assert os.path.isfile(args.video_path), f'Video {args.video_path} does not exist'
    video = cv2.VideoCapture(args.video_path)
    major_ver = cv2.__version__.split('.')[0]  # noqa  # __version__ works, stopping
    assert float(major_ver) >= 3, 'video.get(cv2.CAP_PROP_FPS) needs opencv major version >= 3.'
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dt = 1 / fps  # dt is in seconds
    out = cv2.VideoWriter(filename=args.output_video_path, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                          frameSize=(int(0.5 * width), int(0.5 * height)), isColor=True)
    logger.debug(f'Fps: {fps}, width: {width}, height: {height}')
    logger.debug('Video opened successfully')

    success = True
    frames = []
    sde = SocialDistanceEstimator(dt=dt, img_size=(width, height))
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if args.max_frames_to_process is not None:
        frame_count = min(frame_count, args.max_frames_to_process)
    with tqdm(total=frame_count, desc='Processing frames') as pbar:
        while success and frame_count > pbar.n:
            success, frame = video.read()
            if not success:
                raise ValueError(f'Error reading frame {len(frames) + 1}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = sde(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            if args.display_images:
                cv2.imshow('SocialDistanceEstimation', frame)
                cv2.waitKey(int(dt))
            pbar.update()
    video.release()
    out.release()
    logger.info(f'Finished writing video to {args.output_video_path}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    default_video_path = 'files/videos/OxfordTownCentreDataset.avi'
    # "Oxford Town Centre Dataset" video - Source:
    # https://drive.google.com/file/d/1UMIcffhxGw1aCAyztNWlslHHtayw9Fys/view
    # from https://github.com/DrMahdiRezaei/DeepSOCIAL
    # Dataset:
    # """B. Benfold and I. Reid, "Stable multi-target tracking in real-time surveillance video,"
    # CVPR 2011, 2011, pp. 3457-3464, doi: 10.1109/CVPR.2011.5995667."""

    default_output_video_path = 'files/output.avi'

    parser = argparse.ArgumentParser(description='Social distance estimation.')
    parser.add_argument('--video-path', type=str, default=default_video_path,
                        help='The path to the input video (default: %(default)s)')
    parser.add_argument('--output-video-path', type=str, default=default_output_video_path,
                        help='The path to the output video (default: %(default)s)')
    parser.add_argument('--max-frames-to-process', type=int, default=800,  # TODO: default=None
                        help='Max count of frames of the input video to process.'
                             'If None, all frames are processed (default: %(default)s)')
    parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'ERROR'], default='DEBUG',  # TODO: default='ERROR'
                        help='The logging level of the main script (default: %(default)s)')
    parser.add_argument('--display-images', choices=[True, False], default=False,
                        help='Whether to display processed images (default: %(default)s)')
    # TODO fps, other kwargs

    main(args=parser.parse_args())
