import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.projection.base import Intrinsics, Extrinsics, Camera, normalize, homogeneous_inv, homogeneous
from src.projection.projection import screen2ndc, project


class ProjectionCalibrator:
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
        assert p_top.ndim == 2 and p_top.shape[1] == 2
        assert p_bottom.ndim == 2 and p_bottom.shape[1] == 2
        assert p_top.shape[0] == p_bottom.shape[0]
        assert p_top.shape[0] > 2
        res = self.camera.intrinsics.res
        P_inv = self.camera.intrinsics.proj_inv()
        h = self.person_height

        p_top = screen2ndc(p_top, res)
        p_bottom = screen2ndc(p_bottom, res)
        A = np.cross(p_top, p_bottom)
        v_ = self.__solve_homogeneous_ls__(A)
        normal = P_inv.dot(v_)
        scale = np.linalg.norm(normal)
        normal = normal/scale
        lambdas = np.zeros((p_bottom.shape[0], 2), dtype=float)
        for i, (p_t, p_b) in enumerate(zip(p_top, p_bottom)):
            mtx = np.c_[p_t, -p_b]
            lambdas[i] = np.linalg.lstsq(mtx, h * v_, rcond=None)[0] / scale
        X_mean = np.mean(np.hstack((lambdas[:, 0] * P_inv.dot(p_top.T),
                                    lambdas[:, 1] * P_inv.dot(p_bottom.T))), axis=1)
        self.camera.extrinsics.normal = normal
        self.camera.extrinsics.distance = h/2-normal.dot(X_mean)
        return self.camera


class CalibrationTest(unittest.TestCase):
    def __drawScreen__(self) -> None:
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.p_bottom[:, 0], self.p_bottom[:, 1], c='b')
        ax.scatter(self.p_top[:, 0], self.p_top[:, 1], c='g')
        res = self.camera.intrinsics.res
        plt.xlim([0, res[0]])
        plt.ylim([0, res[1]])
        plt.show()

    def __drawScene__(self) -> None:
        camera = np.array([0, 0, 0, 1], dtype=float)
        look_at = np.array([0, 0, -10000, 1], dtype=float)
        C_inv = self.camera.extrinsics.cam_inv()
        camera = homogeneous_inv(C_inv.dot(camera.T))
        look_at = homogeneous_inv(C_inv.dot(look_at.T))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(camera[0], camera[2], camera[1], s=20, c='y')
        ax.plot([0, camera[0]], [0, camera[2]], [0, camera[1]], c='black')
        ax.plot([camera[0], look_at[0]],
                [camera[2], look_at[2]],
                [camera[1], look_at[1]], c='y')
        ax.scatter(self.top_w[:, 0], self.top_w[:, 2], self.top_w[:, 1], s=10, c='g')
        ax.scatter(self.bottom_w[:, 0], self.bottom_w[:, 2], self.bottom_w[:, 1], s=10, c='b')
        ax.scatter(self.out_w[:, 0], self.out_w[:, 2], self.out_w[:, 1], s=10, c='black')
        ax.scatter(0, 0, 0, c='r', s=50)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.show()

    def __setUpScene__(self, count=1000, noise_strength=500, area=50000, height=1750) -> None:
        self.height = height
        self.bottom_w = (np.random.random((count, 3)).astype(dtype=float) - 0.5) * 2 \
                        * np.array([area, noise_strength, area], dtype=float) \
                        + np.array([area, 0, 0], dtype=float)
        self.top_w = self.bottom_w + height * np.array([0, 1, 0], dtype=float) + \
                     np.random.uniform(-noise_strength, noise_strength, (count, 3)).astype(dtype=float)

    def __clip__(self) -> None:
        zero = np.array([0, 0], dtype=float)
        res = self.camera.intrinsics.res
        mask = np.all(np.logical_and(zero <= self.p_bottom, self.p_bottom < res), axis=1)
        mask = np.logical_and(mask, np.all(np.logical_and(zero <= self.p_top, self.p_top < res), axis=1))
        self.out_w = self.bottom_w[~mask]
        self.bottom_w = self.bottom_w[mask]
        self.top_w = self.top_w[mask]
        self.p_bottom = self.p_bottom[mask]
        self.p_top = self.p_top[mask]
        self.lambda_b = self.lambda_b[mask]
        self.lambda_t = self.lambda_t[mask]

    def __setUpCamera__(self) -> None:
        self.camera = Camera(extrinsics=Extrinsics(normal=normalize(np.array([0.1, 1, 0.5], dtype=float)),
                                                   distance=20000),
                             intrinsics=Intrinsics(cx=0,
                                                   cy=0,
                                                   fx=1.5,
                                                   fy=1.5,
                                                   res=np.array([800, 600], dtype=float)))

    def setUp(self) -> None:
        print('Setup')
        self.verbose = True
        self.__setUpCamera__()
        self.__setUpScene__()
        self.p_bottom, self.lambda_b = project(self.bottom_w, self.camera)
        self.p_top, self.lambda_t = project(self.top_w, self.camera)
        self.__clip__()
        self.calibrator = ProjectionCalibrator(self.camera.intrinsics, self.height)

    def test_calibration(self) -> None:
        print('Test')
        if self.verbose:
            self.__drawScene__()
            self.__drawScreen__()
        camera = self.calibrator.calibrate(p_bottom=self.p_bottom, p_top=self.p_top)
        tolerance = 1e-2
        self.assertGreater(tolerance, np.linalg.norm(np.cross(self.camera.extrinsics.normal, camera.extrinsics.normal)))
        self.assertAlmostEqual(1, self.camera.extrinsics.distance/camera.extrinsics.distance, delta=tolerance)


if __name__ == '__main__':
    unittest.main()
