import numpy as np

from src.projection.base import Camera


class ProjectionCalibrator:
    @staticmethod
    def __validate_input__(p_top: np.ndarray, p_bottom: np.ndarray) -> None:
        assert p_top.ndim == 2 and p_top.shape[1] == 2
        assert p_bottom.ndim == 2 and p_bottom.shape[1] == 2
        assert p_top.shape[0] == p_bottom.shape[0]
        assert p_top.shape[0] > 2

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        pass
