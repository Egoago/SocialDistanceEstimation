import numpy as np
from skimage.measure import ransac

from src.projection.base import Intrinsics, Camera
from src.projection.calibrator import ProjectionCalibrator
from src.projection.calibrators.linear import LinearCalibrator
from src.projection.projection import back_project, project


class RansacCalibrator(ProjectionCalibrator):
    class CalibrationWrapper:
        calibrator: ProjectionCalibrator

        def residuals(self, pixels: np.ndarray) -> np.ndarray:
            pixels_b = pixels[:, -2:]
            pixels_t = pixels[:, :-2]
            camera = self.calibrator.camera
            points = back_project(pixels_b, camera)
            points[1] += self.calibrator.person_height
            pixels_t_new, scaling_factors = project(points, camera)
            pixel_diff = pixels_t_new - pixels_t
            rel_diff = pixel_diff/camera.intrinsics.res
            res = np.linalg.norm(rel_diff, axis=-1)
            d = camera.extrinsics.distance*camera.extrinsics.normal.dot(np.array([0, 0, 1], dtype=float))
            res = res * (np.abs(scaling_factors)) / d
            med = np.median(res)
            avg = np.average(res)
            dev = np.std(res)
            return res

        def estimate(self, pixels: np.ndarray) -> bool:
            self.calibrator.calibrate(pixels[:, :-2], pixels[:, -2:])
            return True

    def __init__(self, calibrator: ProjectionCalibrator):
        self.CalibrationWrapper.calibrator = calibrator
        self.camera = self.CalibrationWrapper.calibrator.camera
        self.inliers = np.empty(0, dtype=float)

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        self.__validate_input__(p_top, p_bottom)
        data = np.hstack((p_top, p_bottom))
        _, self.inliers = ransac(data=data,
                                 min_samples=5,
                                 residual_threshold=0.1,
                                 max_trials=50,
                                 model_class=RansacCalibrator.CalibrationWrapper)
        if self.inliers is not None:
            self.camera = self.CalibrationWrapper.calibrator.calibrate(p_top[self.inliers], p_bottom[self.inliers])
        return self.camera
