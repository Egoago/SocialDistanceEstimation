import numpy as np

from src.projection.base import Camera


class ProjectionCalibrator:
    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        pass
