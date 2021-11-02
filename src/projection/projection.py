import unittest
from typing import Tuple, Any

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
    if ndc.ndim == 2:
        depth = ndc[:, 2]
    elif ndc.ndim == 1:
        depth = ndc[2]
    return ndc2screen(ndc, camera.intrinsics.res), depth


def back_project(pixels: np.ndarray, camera: Camera = Camera(), scaling_factors: np.ndarray = None) -> np.ndarray:
    ndc = screen2ndc(pixels, camera.intrinsics.res)
    P_inv = camera.intrinsics.proj_inv()
    C_inv = camera.extrinsics.cam_inv()

    if scaling_factors is not None:
        if ndc.ndim == 2:
            points_cam = P_inv.dot(ndc.T).T * scaling_factors[:, None]
        elif ndc.ndim == 1:
            points_cam = P_inv.dot(ndc) * scaling_factors
    else:
        points_cam = P_inv.dot(ndc.T)
        with np.errstate(divide='ignore'):
            if points_cam.ndim == 2:
                points_cam = -camera.extrinsics.distance*points_cam.T/camera.extrinsics.normal.dot(points_cam)[:, None]
            elif points_cam.ndim == 1:
                points_cam = -camera.extrinsics.distance*points_cam.T/camera.extrinsics.normal.dot(points_cam)
    return homogeneous_inv(C_inv.dot(homogeneous(points_cam).T).T)


class ProjectorTest(unittest.TestCase):
    @staticmethod
    def __WorldGrid__(resolution=10, size=100000) -> np.ndarray:
        points = []
        for x in np.linspace(-size, size, resolution):
            for z in np.linspace(-size, size, resolution):
                points.append([x, 0, z])
        return np.array(points, dtype=float)

    def __drawCamera(self, ax: Any):
        camera = np.array([0, 0, 0, 1], dtype=float)
        look_at = np.array([0, 0, -10000, 1], dtype=float)
        camera = self.camera.extrinsics.cam_inv().dot(camera.T)
        look_at = self.camera.extrinsics.cam_inv().dot(look_at.T)
        camera = homogeneous_inv(camera)
        look_at = homogeneous_inv(look_at)
        ax.scatter(camera[0], camera[2], camera[1], s=20, c='r')
        ax.plot([0, camera[0]], [0, camera[2]], [0, camera[1]], c='black')
        ax.plot([camera[0], look_at[0]],
                [camera[2], look_at[2]],
                [camera[1], look_at[1]], c='r')

    def __drawWorld__(self, points, c='b', last=True, ax=None) -> Any:
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        if points.ndim == 2:
            ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=c)
        elif points.ndim == 1:
            ax.scatter(points[0], points[2], points[1], c=c)
        if last:
            self.__drawCamera(ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        return ax

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
        self.verbose = True
        self.camera = Camera()

    def test_back_project(self) -> None:
        # something is not right...
        pixels = self.__ScreenGrid__()
        points_w = back_project(pixels, self.camera)
        if self.verbose:
            world_points = self.__WorldGrid__()
            ax = self.__drawWorld__(world_points, c='b', last=False)
            self.__drawWorld__(points_w, c='r', ax=ax)

    def test_project(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = project(world_points, self.camera)
        if self.verbose:
            ax = self.__drawWorld__(world_points[depths < 0], c='g', last=False)
            self.__drawWorld__(world_points[depths > 0], c='r', ax=ax)
            self.__drawScreen__(pixels[depths > 0])

    def test_all_with_scaling(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = project(world_points, self.camera)
        world_points_new = back_project(pixels, self.camera, depths)
        if self.verbose:
            ax = self.__drawWorld__(world_points, c='g', last=False)
            self.__drawWorld__(world_points_new, c='r', ax=ax)
        self.assertTrue(np.allclose(world_points, world_points_new))

    def test_all_without_scaling(self) -> None:
        world_points = self.__WorldGrid__()
        pixels, depths = project(world_points, self.camera)
        world_points_new = back_project(pixels, self.camera)
        if self.verbose:
            ax = self.__drawWorld__(world_points, c='g', last=False)
            self.__drawWorld__(world_points_new, c='r', ax=ax)
        self.assertTrue(np.allclose(world_points, world_points_new))

    def test_one_without_scaling(self) -> None:
        world_points = self.__WorldGrid__()
        point_w = np.array([50000, 5000, -50000])
        pixels, depths = project(point_w, self.camera)
        point_w_new = back_project(pixels, self.camera)
        if self.verbose:
            ax = self.__drawWorld__(world_points, c='b', last=False)
            ax = self.__drawWorld__(point_w, c='g', ax=ax, last=False)
            self.__drawWorld__(point_w_new, c='r', ax=ax)
        self.assertLess(point_w_new.dot(self.camera.extrinsics.normal)-self.camera.extrinsics.distance, 1e-4)

    def test_one_with_scaling(self) -> None:
        point_w = np.array([50000, 5000, -50000])
        world_points = self.__WorldGrid__()
        pixels, depths = project(point_w, self.camera)
        point_w_new = back_project(pixels, self.camera, depths)
        if self.verbose:
            ax = self.__drawWorld__(world_points, c='b', last=False)
            ax = self.__drawWorld__(point_w, c='g', ax=ax, last=False)
            self.__drawWorld__(point_w_new, c='r', ax=ax)
        self.assertTrue(np.allclose(point_w, point_w_new))


if __name__ == '__main__':
    unittest.main()
