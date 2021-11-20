import numpy as np

from src.projection.base import Camera, Intrinsics, Extrinsics


class ProjectionCalibrator:
    def __init__(self, intrinsics: Intrinsics, person_height: float = 1750):
        self.person_height = person_height
        self.camera = Camera(intrinsics=intrinsics,
                             extrinsics=Extrinsics(distance=10000,
                                                   normal=np.array([0.5, 1, 0.1], dtype=float)))

    @staticmethod
    def __validate_input__(p_top: np.ndarray, p_bottom: np.ndarray) -> None:
        assert p_top.ndim == 2 and p_top.shape[1] == 2
        assert p_bottom.ndim == 2 and p_bottom.shape[1] == 2
        assert p_top.shape[0] == p_bottom.shape[0]
        assert p_top.shape[0] > 2

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        pass
