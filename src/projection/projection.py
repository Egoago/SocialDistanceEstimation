import unittest
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.projection.base import Camera, homogeneous, homogeneous_inv


class Projector:
    def __init__(self, camera: Camera = Camera()):
        self.camera = camera

    def screen2ndc(self, points: np.ndarray) -> np.ndarray:
        return homogeneous(points / self.camera.intrinsics.res * 2 - 1)

    def ndc2screen(self, ndc: np.ndarray) -> np.ndarray:
        ndc = -homogeneous_inv(ndc)  # perspective divide
        dc = (ndc + 1) / 2          # ndc to dc
        return dc * self.camera.intrinsics.res  # dc to screen

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = self.camera.intrinsics.proj()
        R, T = self.camera.extrinsics.cam()

        points_cam = (np.c_[R, T].dot(homogeneous(points).T)).T
        ndc = P.dot(points_cam.T).T
        depth = ndc[:, 2]
        return self.ndc2screen(ndc), depth

    def back_project(self, pixels: np.ndarray, scaling_factors: np.ndarray) -> np.ndarray:
        ndc = self.screen2ndc(pixels)
        P_inv = self.camera.intrinsics.proj_inv()
        R_inv, T_inv = self.camera.extrinsics.cam_inv()

        points_cam = P_inv.dot(ndc.T).T * scaling_factors[:, None]
        return (np.c_[R_inv, T_inv].dot(homogeneous(points_cam).T)).T


class ProjectorTest(unittest.TestCase):
    @staticmethod
    def __WorldGrid__(resolution=10, size=10000) -> np.ndarray:
        points = []
        for x in np.linspace(-size, size, resolution):
            for z in np.linspace(-size, size, resolution):
                points.append([x, 0, z])
        return np.array(points, dtype=float)

    def __drawWorld__(self, points):
        camera = self.projector.camera.extrinsics.cam_inv()[1]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1])
        ax.scatter(camera[0], camera[2], camera[1], s=20, c='r')
        ax.plot([0, camera[0]], [0, camera[2]], [0, camera[1]])
        plt.show()

    def __ScreenGrid__(self, resolution=10) -> np.ndarray:
        res = self.projector.camera.intrinsics.res
        pixels = []
        for x in np.linspace(0, res[0], resolution):
            for y in np.linspace(0, res[1], resolution):
                pixels.append([x, y])
        return  np.array(pixels, dtype=float)

    @staticmethod
    def __drawScreen__(pixels):
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pixels[:, 0], pixels[:, 1])
        plt.show()

    def setUp(self) -> None:
        self.verbose = True
        self.projector = Projector()

    def test_back_project(self) -> None:
        pixels = self.__ScreenGrid__()
        world_points = self.projector.back_project(pixels, 1000*np.ones(pixels.shape[0]))
        if self.verbose:
            self.__drawScreen__(pixels)
            self.__drawWorld__(world_points)

    def test_project(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = self.projector.project(world_points)
        if self.verbose:
            self.__drawWorld__(world_points)
            self.__drawScreen__(pixels)

    def test_all(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = self.projector.project(world_points)
        world_points_new = self.projector.back_project(pixels, depths)
        if self.verbose:
            self.__drawWorld__(world_points-world_points_new)


if __name__ == '__main__':
    unittest.main()
