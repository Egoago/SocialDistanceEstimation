import numpy as np

from src.projection.base import Intrinsics, Extrinsics, Camera
from src.projection.calibrator import ProjectionCalibrator
from src.projection.projection import screen2ndc


class LinearCalibrator(ProjectionCalibrator):
    def __init__(self, intrinsics: Intrinsics, person_height: float = 1750):
        self.person_height = person_height
        self.camera = Camera(intrinsics=intrinsics,
                             extrinsics=Extrinsics(distance=0,
                                                   normal=np.array([0, 1, 0], dtype=float)))

    @staticmethod
    def __solve_homogeneous_ls__(mtx: np.ndarray) -> np.ndarray:
        eigen_pairs = np.linalg.eig(np.dot(mtx.T, mtx))
        min_index = np.argmin(eigen_pairs[0])
        return eigen_pairs[1][:, min_index]

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        self.__validate_input__(p_top, p_bottom)
        res = self.camera.intrinsics.res
        P_inv = self.camera.intrinsics.proj_inv()
        h = self.person_height

        p_top = screen2ndc(p_top, res)
        p_bottom = screen2ndc(p_bottom, res)
        A = np.cross(p_top, p_bottom)
        v_ = self.__solve_homogeneous_ls__(A)
        normal = P_inv.dot(v_)
        scale = np.linalg.norm(normal)
        normal = normal / scale
        lambdas = np.zeros((p_bottom.shape[0], 2), dtype=float)
        for i, (p_t, p_b) in enumerate(zip(p_top, p_bottom)):
            mtx = np.c_[p_t, -p_b]
            lambdas[i] = np.linalg.lstsq(mtx, h * v_, rcond=None)[0] / scale
        X_mean = np.mean(np.hstack((lambdas[:, 0] * P_inv.dot(p_top.T),
                                    lambdas[:, 1] * P_inv.dot(p_bottom.T))), axis=1)
        self.camera.extrinsics.normal = normal
        self.camera.extrinsics.distance = h / 2 - normal.dot(X_mean)
        return self.camera

