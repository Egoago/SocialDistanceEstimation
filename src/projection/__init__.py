from .base import Camera, Intrinsics
from .calibrator import ProjectionCalibrator
from .projection import back_project, project, opencv2opengl, opengl2opencv


def create_calibrator(intrinsics: Intrinsics,
                      use_ransac=False,
                      person_height=1720.0,
                      method='linear') -> ProjectionCalibrator:
    _calibrator = None
    if method == 'linear':
        from .calibrators.linear import LinearCalibrator
        _calibrator = LinearCalibrator(intrinsics, person_height)
    elif method == 'least_squares':
        from .calibrators.least_squares import LeastSquaresCalibrator
        _calibrator = LeastSquaresCalibrator(intrinsics, person_height)
    if use_ransac:
        from .calibrators.ransac import RansacCalibrator
        return RansacCalibrator(_calibrator)
    else:
        return _calibrator

