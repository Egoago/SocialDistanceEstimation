import unittest
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.projection.base import Intrinsics, Extrinsics, Camera, normalize


class ProjectionCalibrator:
    def __init__(self, intrinsics: Intrinsics, person_height: float = 1750):
        self.K = np.array([[intrinsics.fx, 0, intrinsics.cx],
                           [0, intrinsics.fy, intrinsics.cy],
                           [0, 0, 1]], dtype=float)
        self.K_inv = np.linalg.inv(self.K)
        self.person_height = person_height
        self.camera = Camera(intrinsics=intrinsics,
                             extrinsics=Extrinsics(distance=0,
                                                   normal=np.array([0, 1, 0], dtype=float)))

    @staticmethod
    def __solve_homogeneous_ls__(mtx: np.ndarray) -> np.ndarray:
        eigen_pairs = np.linalg.eig(np.dot(mtx.T, mtx))
        min_index = np.argmin(eigen_pairs[0])
        return eigen_pairs[1][:, min_index]

    def __screen2ndc__(self, points: np.ndarray) -> np.ndarray:
        return np.c_[points / self.camera.intrinsics.res * 2 - 1,
                     np.ones(points.shape[0])]

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        assert p_top.ndim == 2 and p_top.shape[1] == 2
        assert p_bottom.ndim == 2 and p_bottom.shape[1] == 2
        assert p_top.shape[0] == p_bottom.shape[0]
        count = p_bottom.shape[0]
        h = self.person_height
        p_top = self.__screen2ndc__(p_top)
        p_bottom = self.__screen2ndc__(p_bottom)
        A = np.cross(p_top, p_bottom)
        v_ = self.__solve_homogeneous_ls__(A)
        # print('v_', v_)
        # print('|v_|', np.linalg.norm(v_))
        normal = self.K_inv.dot(v_)
        scale = np.linalg.norm(normal)
        normal = normalize(normal)
        lambdas = np.zeros((count, 2), dtype=float)  # lambda bottoms
        for i, (p_t, p_b) in enumerate(zip(p_top, p_bottom)):
            mtx = np.column_stack((p_t, -p_b))
            lambdas[i] = np.linalg.lstsq(mtx, h * v_, rcond=None)[0]
        # print('scale', scale)
        lambdas = lambdas / scale
        X_t_avg = np.mean(lambdas[:, 0] * self.K_inv.dot(p_top.T), axis=1)
        # print('X_t_avg', X_t_avg)
        X_b_avg = np.mean(lambdas[:, 1] * self.K_inv.dot(p_bottom.T), axis=1)
        # print
        self.camera.extrinsics.normal = normal
        self.camera.extrinsics.distance = 0.5 * (h - normal.dot(X_t_avg + X_b_avg))
        print('normal', self.camera.extrinsics.normal)
        print('distance', self.camera.extrinsics.distance)
        return self.camera

    def matrix(self) -> np.ndarray:
        pass


class CalibrationTest(unittest.TestCase):
    @staticmethod
    def noise(size, amplitude: float) -> np.ndarray:
        return (np.random.rand(*size) - 0.5) * 2 * amplitude

    def __drawScreen__(self) -> None:
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.p_bottom[:, 0], self.p_bottom[:, 1])
        res = self.camera.intrinsics.res
        plt.xlim([0, res[0]])
        plt.ylim([0, res[1]])
        plt.show()

    def __drawScene__(self) -> None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.top[:, 0], self.top[:, 2], self.top[:, 1], s=10, c='g')
        ax.scatter(self.bottom[:, 0], self.bottom[:, 2], self.bottom[:, 1], s=10)
        ax.scatter(0, 0.0, 0, c='r', s=50)
        plt.show()

    def __setUpScene__(self, count=500, noise_strength=500) -> None:
        normal = self.camera.extrinsics.normal
        d = self.camera.extrinsics.distance
        horizontal = np.array([0, 0, 1], dtype=float)
        a = normalize(np.cross(normal, horizontal))
        base = np.array([a, normalize(np.cross(a, normal))], dtype=float)
        data = np.matmul(self.noise((count, 2), 50000), base)
        self.bottom = data - d * normal + self.noise((count, 3), noise_strength)
        self.height = 1750
        self.top = self.bottom + self.height * normal + self.noise((count, 3), noise_strength)

    def __project__(self, points) -> Tuple[np.ndarray, np.ndarray]:
        i = self.camera.intrinsics
        P = np.array([[i.fx, 0, i.cx],
                      [0, i.fy, i.cy],
                      [0, 0, 1]], dtype=float)
        res = self.camera.intrinsics.res
        pixels = np.matmul(points, P.T)
        depth = pixels[:, 2]
        pixels = pixels[:, :-1] / depth[:, None]  # perspective divide
        pixels = (pixels + 1) / 2  # ndc to dc
        pixels = pixels * res  # dc to screen
        return pixels, depth

    def __clip__(self) -> None:
        zero = np.array([0, 0], dtype=float)
        res = self.camera.intrinsics.res
        mask = np.all(np.logical_and(zero <= self.p_bottom, self.p_bottom < res), axis=1)
        mask = np.logical_and(mask, np.all(np.logical_and(zero <= self.p_top, self.p_top < res), axis=1))
        self.bottom = self.bottom[mask]
        self.top = self.top[mask]
        self.p_bottom = self.p_bottom[mask]
        self.p_top = self.p_top[mask]
        self.lambda_b = self.lambda_b[mask]
        self.lambda_t = self.lambda_t[mask]

    def __setUpCamera__(self) -> None:
        self.camera = Camera(extrinsics=Extrinsics(normal=normalize(np.array([0.1, -1, 1], dtype=float)),
                                                   distance=20000),
                             intrinsics=Intrinsics(cx=0,
                                                   cy=0,
                                                   fx=2.5,
                                                   fy=2.5,
                                                   res=np.array([800, 600], dtype=float)))

    def setUp(self) -> None:
        print('Setup')
        self.__setUpCamera__()
        self.__setUpScene__()
        self.p_bottom, self.lambda_b = self.__project__(self.bottom)
        self.p_top, self.lambda_t = self.__project__(self.top)
        self.__clip__()
        self.__drawScene__()
        self.__drawScreen__()
        self.calibrator = ProjectionCalibrator(self.camera.intrinsics, self.height)
        print('normal', self.camera.extrinsics.normal)
        print('distance', self.camera.extrinsics.distance)

    def test_something(self) -> None:
        self.assertEqual(True, True)  # add assertion here
        print('Test')
        self.calibrator.calibrate(p_bottom=self.p_bottom, p_top=self.p_top)


if __name__ == '__main__':
    unittest.main()
