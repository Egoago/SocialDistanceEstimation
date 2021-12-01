import json
import logging
import os.path
import time

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

    zoom = 1
    left_factor = 1
    top_factor = 1
    if args.focus != '':
        zoom = 1.5
        # TODO check center of attention with zoom != 2
        center_of_attention = dict(top=(1, 0), bottom=(1, 2), left=(0, 1), right=(2, 1), bottomright=(2, 2))
        left_factor, top_factor = center_of_attention[args.focus]

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / zoom)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / zoom)
    left = left_factor * int(video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2 - width / 2)
    top = top_factor * int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 - height / 2)
    dt = 1 / fps  # dt is in seconds
    out_frame_shape = 540, 960
    frameSize = out_frame_shape[::-1]  # cv2 VideoWriter expects width x height  # noqa
    out = cv2.VideoWriter(filename=args.output_video_path, fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                          frameSize=frameSize, isColor=True)
    logger.debug(f'Fps: {fps}, scaled width: {width}, scaled height: {height}')
    logger.debug('Video opened successfully')

    success = True
    frames = []
    sde = SocialDistanceEstimator(dt=dt, input_shape=(width, height), output_shape=out_frame_shape)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if args.max_frames_to_process is not None:
        frame_count = min(frame_count, args.max_frames_to_process)
    if args.skip_frames is not None and args.skip_frames > 0:
        logger.debug(f'Skipping {args.skip_frames} frames')
        i = 0
        while i < args.skip_frames:
            success, frame = video.read()
            i += 1
    with tqdm(total=frame_count, desc='Processing frames') as bar:
        while success and frame_count > bar.n:
            frame_start = time.time_ns()
            success, frame = video.read()
            if not success:
                raise ValueError(f'Error reading frame {len(frames) + 1}')
            frame = frame[top:top+height, left:left+width]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = sde(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            assert frame.shape[:2] == out_frame_shape
            out.write(frame)
            if args.display_images:
                cv2.imshow('SocialDistanceEstimation', frame)
                frame_end = time.time_ns()
                adt = (frame_end - frame_start)/1e9  # adt in seconds
                factor = 0.0
                t = factor * (dt - adt) * 1e3  # t in milliseconds
                if t < 1:
                    t = 1
                cv2.waitKey(int(t))
            bar.update()
    video.release()
    out.release()
    logger.info(f'Finished writing video to {args.output_video_path}')
    cv2.destroyAllWindows()
    logger.debug('Exit')


if __name__ == '__main__':
    import argparse

    # Other videos: [
    # 'files/videos/20211129_123707.mp4',
    # 'files/videos/20211129_124248.mp4',
    # 'files/videos/20211129_152215.mp4',
    # 'files/videos/20211129_153727.mp4']

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
    parser.add_argument('--max-frames-to-process', type=int, default=2000,  # TODO: default=None
                        help='Max count of frames of the input video to process.'
                             'If None, all frames are processed (default: %(default)s)')
    parser.add_argument('--skip-frames', type=int, default=None,
                        help='Amount of frames to skip from the start of the input video (default: %(default)s)')
    parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'ERROR'], default='DEBUG',  # TODO: default='ERROR'
                        help='The logging level of the main script (default: %(default)s)')
    parser.add_argument('--display-images', choices=[True, False], default=False,
                        help='Whether to display processed images (default: %(default)s)')
    parser.add_argument('--focus', choices=['top', 'bottom', 'left', 'right', 'bottomright', ''], default='',
                        help='Whether to focus on a particular section of the input (default: %(default)s)')
    # TODO fps, other kwargs

    main(args=parser.parse_args())
