import numpy as np
from skimage.measure import ransac

from src.projection.base import Intrinsics, Camera
from src.projection.calibrator import ProjectionCalibrator
from src.projection.calibrators.linear import LinearCalibrator
from src.projection.projection import back_project, project


class RansacCalibrator(ProjectionCalibrator):
    class CalibrationWrapper:
        calibrator: LinearCalibrator

        def residuals(self, pixels: np.ndarray) -> np.ndarray:
            pixels_b = pixels[:, -2:]
            pixels_t = pixels[:, :-2]
            points = back_project(pixels_b, self.calibrator.camera)
            points[1] += self.calibrator.person_height
            pixels_t_new, _ = project(points, self.calibrator.camera)
            return np.linalg.norm(pixels_t_new - pixels_t, axis=1)

        def estimate(self, pixels: np.ndarray) -> bool:
            self.calibrator.calibrate(pixels[:, :-2], pixels[:, -2:])
            return True

    def __init__(self, intrinsics: Intrinsics, person_height: float = 1750):
        self.CalibrationWrapper.calibrator = LinearCalibrator(intrinsics, person_height)
        self.camera = self.CalibrationWrapper.calibrator.camera
        self.inliers = np.empty(0, dtype=float)

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        data = np.hstack((p_top, p_bottom))
        _, self.inliers = ransac(data=data,
                                 min_samples=3,
                                 residual_threshold=15,
                                 max_trials=50,
                                 model_class=RansacCalibrator.CalibrationWrapper)
        self.camera = self.CalibrationWrapper.calibrator.calibrate(p_top[self.inliers], p_bottom[self.inliers])
        return self.camera
