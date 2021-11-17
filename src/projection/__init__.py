from .base import Camera, Intrinsics
from .calibrator import ProjectionCalibrator
from .projection import back_project, project, opencv2opengl, opengl2opencv


def create_calibrator(intrinsics: Intrinsics, use_ransac=False, person_height=1720.0) -> ProjectionCalibrator:
    if use_ransac:
        from calibrators import RansacCalibrator
        return RansacCalibrator(intrinsics, person_height)
    else:
        from calibrators import LinearCalibrator
        return LinearCalibrator(intrinsics, person_height)
