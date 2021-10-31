import unittest
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.projection.base import Camera, homogeneous, homogeneous_inv


def screen2ndc(points: np.ndarray, res: np.ndarray) -> np.ndarray:
    return homogeneous(points / res * 2 - 1)


def ndc2screen(ndc: np.ndarray, res: np.ndarray) -> np.ndarray:
    ndc = homogeneous_inv(ndc)  # perspective divide
    dc = (ndc + 1) / 2  # ndc to dc
    return dc * res  # dc to screen


def project(points: np.ndarray, camera: Camera = Camera()) -> Tuple[np.ndarray, np.ndarray]:
    P = camera.intrinsics.proj()
    C = camera.extrinsics.cam()

    points_cam = homogeneous_inv(C.dot(homogeneous(points).T).T)
    ndc = P.dot(points_cam.T).T
    depth = ndc[:, 2]
    return ndc2screen(ndc, camera.intrinsics.res), depth


def back_project(pixels: np.ndarray, scaling_factors: np.ndarray, camera: Camera = Camera()) -> np.ndarray:
    ndc = screen2ndc(pixels, camera.intrinsics.res)
    P_inv = camera.intrinsics.proj_inv()
    C_inv = camera.extrinsics.cam_inv()

    points_cam = P_inv.dot(ndc.T).T * scaling_factors[:, None]
    return homogeneous_inv(C_inv.dot(homogeneous(points_cam).T).T)


class ProjectorTest(unittest.TestCase):
    @staticmethod
    def __WorldGrid__(resolution=10, size=100000) -> np.ndarray:
        points = []
        for x in np.linspace(-size, size, resolution):
            for z in np.linspace(-size, size, resolution):
                points.append([x, 0, z])
        return np.array(points, dtype=float)

    def __drawWorld__(self, points, c='b'):
        camera = np.array([0, 0, 0, 1], dtype=float)
        lookat = np.array([0, 0, -10000, 1], dtype=float)
        camera = self.camera.extrinsics.cam_inv().dot(camera.T)
        lookat = self.camera.extrinsics.cam_inv().dot(lookat.T)
        camera = homogeneous_inv(camera)
        lookat = homogeneous_inv(lookat)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=c)
        ax.scatter(camera[0], camera[2], camera[1], s=20, c='r')
        ax.plot([0, camera[0]], [0, camera[2]], [0, camera[1]], c='black')
        ax.plot([camera[0], lookat[0]],
                [camera[2], lookat[2]],
                [camera[1], lookat[1]], c='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def __ScreenGrid__(self, resolution=10) -> np.ndarray:
        res = self.camera.intrinsics.res
        pixels = []
        for x in np.linspace(0, res[0], resolution):
            for y in np.linspace(0, res[1], resolution):
                pixels.append([x, y])
        return np.array(pixels, dtype=float)

    @staticmethod
    def __drawScreen__(pixels):
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pixels[:, 0], pixels[:, 1])
        plt.show()

    def setUp(self) -> None:
        self.verbose = False
        self.camera = Camera()

    def test_back_project(self) -> None:
        pixels = self.__ScreenGrid__()
        world_points = back_project(pixels, 1000 * np.ones(pixels.shape[0]), self.camera)
        if self.verbose:
            self.__drawScreen__(pixels)
            self.__drawWorld__(world_points)

    def test_project(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = project(world_points, self.camera)
        if self.verbose:
            self.__drawWorld__(world_points[depths < 0], c='r')
            self.__drawWorld__(world_points[depths > 0], c='g')
            self.__drawScreen__(pixels[depths > 0])

    def test_all(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = project(world_points, self.camera)
        world_points_new = back_project(pixels, depths, self.camera)
        if self.verbose:
            self.__drawWorld__(world_points, c='r')
            self.__drawWorld__(world_points_new, c='g')
        self.assertTrue(np.allclose(world_points, world_points_new))


if __name__ == '__main__':
    unittest.main()
