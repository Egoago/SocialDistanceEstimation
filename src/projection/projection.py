import unittest
from typing import Tuple
import numpy as np

from .base import Camera, homogeneous, homogeneous_inv
from .calibrators.test.drawing import draw_3d_points, draw_camera


def screen2ndc(points: np.ndarray, res: np.ndarray) -> np.ndarray:
    return homogeneous(points / res * 2 - 1)


def ndc2screen(ndc: np.ndarray, res: np.ndarray) -> np.ndarray:
    ndc = homogeneous_inv(ndc)  # perspective divide
    dc = (ndc + 1) / 2  # ndc to dc
    return dc * res  # dc to screen


def opencv2opengl(coords: Tuple[float, float], img_height: int) -> Tuple[float, float]:
    return coords[0], img_height-coords[1]


def opengl2opencv(coords: Tuple[float, float], img_height: int) -> Tuple[float, float]:
    return opencv2opengl(coords, img_height)


def project(points: np.ndarray, camera: Camera = Camera()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points to screen pixels.

    :param points: 3D points (in mms) to be projected on the screen
    :param camera: the calibrated camera to use
    :return: Tuple of numpy arrays containing the projected screen pixels
    """
    P = camera.intrinsics.proj()
    C = camera.extrinsics.cam()

    points_cam = homogeneous_inv(C.dot(homogeneous(points).T).T)
    ndc = P.dot(points_cam.T).T
    depth = np.zeros(ndc.shape[0], dtype=float)
    if ndc.ndim == 2:
        depth = ndc[:, 2]
    elif ndc.ndim == 1:
        depth = ndc[2]
    return ndc2screen(ndc, camera.intrinsics.res), depth


def back_project(pixels: np.ndarray, camera: Camera = Camera(), scaling_factors: np.ndarray = None) -> np.ndarray:
    """
    Back-projects screen pixels to 3D points.

    :param pixels: Screen pixels to back-project
    :param scaling_factors: scaling factors to use during back-projection.
    If all is 1, back-projected 3D coordinates will be the screen pixels' coordinates in world space.
    If None (default) the scaling factors will be chosen to project on to the `y=0` ground plane.
    :param camera: the calibrated camera to use
    :return: numpy array containing the back-projected 3D coordinates
    """
    ndc = screen2ndc(pixels, camera.intrinsics.res)
    P_inv = camera.intrinsics.proj_inv()
    C_inv = camera.extrinsics.cam_inv()
    points_cam = np.zeros(ndc.shape[0], dtype=float)
    if scaling_factors is not None:
        if ndc.ndim == 2:
            points_cam = P_inv.dot(ndc.T).T * scaling_factors[:, None]
        elif ndc.ndim == 1:
            points_cam = P_inv.dot(ndc) * scaling_factors
    else:
        points_cam = P_inv.dot(ndc.T)
        with np.errstate(divide='ignore'):
            if points_cam.ndim == 2:
                points_cam = -camera.extrinsics.distance*points_cam.T/(camera.extrinsics.normal.dot(points_cam)+1e-16)[:, None]
            elif points_cam.ndim == 1:
                points_cam = -camera.extrinsics.distance*points_cam.T/camera.extrinsics.normal.dot(points_cam)
    return homogeneous_inv(C_inv.dot(homogeneous(points_cam).T).T)


class ProjectorTest(unittest.TestCase):
    @staticmethod
    def __WorldGrid__(resolution=10, size=10000) -> np.ndarray:
        points = []
        for x in np.linspace(-size, size, resolution):
            for z in np.linspace(-size, size, resolution):
                points.append([x, 0, z])
        return np.array(points, dtype=float)

    def __ScreenGrid__(self, resolution=10) -> np.ndarray:
        res = self.camera.intrinsics.res
        pixels = []
        for x in np.linspace(0, res[0], resolution):
            for y in np.linspace(0, res[1], resolution):
                pixels.append([x, y])
        return np.array(pixels, dtype=float)

    def setUp(self) -> None:
        self.verbose = True
        self.camera = Camera()
        self.world_points = self.__WorldGrid__()
        self.pixels = self.__ScreenGrid__()

    def test_back_project(self) -> None:
        points_w = back_project(self.pixels, self.camera)
        if self.verbose:
            ax = draw_3d_points(self.world_points, c='b', last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(points_w, c='r', ax=ax)

    def test_project(self) -> None:
        pixels, depths = project(self.world_points, self.camera)
        if self.verbose:
            ax = draw_3d_points(self.world_points[depths < 0], c='b', last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(self.world_points[depths > 0], c='r', last=False)

    def test_all_with_scaling(self) -> None:
        pixels, depths = project(self.world_points, self.camera)
        world_points_new = back_project(pixels, self.camera, depths)
        if self.verbose:
            ax = draw_3d_points(self.world_points, c='g', last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(world_points_new, c='r', ax=ax)
        self.assertTrue(np.allclose(self.world_points, world_points_new))

    def test_all_without_scaling(self) -> None:
        pixels, depths = project(self.world_points, self.camera)
        world_points_new = back_project(pixels, self.camera)
        if self.verbose:
            ax = draw_3d_points(self.world_points, c='g', last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(world_points_new, c='r', ax=ax)
        self.assertTrue(np.allclose(self.world_points, world_points_new))

    def test_one_without_scaling(self) -> None:
        point_w = np.array([5000, 5000, -5000])
        pixels, depths = project(point_w, self.camera)
        point_w_new = back_project(pixels, self.camera)
        if self.verbose:
            ax = draw_3d_points(self.world_points, c='b', last=False)
            ax = draw_3d_points(point_w, c='g', ax=ax, last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(point_w_new, c='r', ax=ax)
        self.assertLess(point_w_new.dot(self.camera.extrinsics.normal)-self.camera.extrinsics.distance, 1e-4)

    def test_one_with_scaling(self) -> None:
        point_w = np.array([5000, 5000, -5000])
        pixels, depths = project(point_w, self.camera)
        point_w_new = back_project(pixels, self.camera, depths)
        if self.verbose:
            ax = draw_3d_points(self.world_points, c='b', last=False)
            ax = draw_3d_points(point_w, c='g', ax=ax, last=False)
            draw_camera(self.camera, ax)
            draw_3d_points(point_w_new, c='r', ax=ax)
        self.assertTrue(np.allclose(point_w, point_w_new))


if __name__ == '__main__':
    unittest.main()
