import unittest

import numpy as np

from src.projection.base import Camera, normalize, Extrinsics, Intrinsics
from src.projection.calibrators.linear import LinearCalibrator
from src.projection.calibrators.ransac import RansacCalibrator
from src.projection.calibrators.test.drawing import draw_2d_points, draw_3d_points, draw_camera
from src.projection.projection import project


class LinearTest(unittest.TestCase):
    def __setUpScene__(self, count=2000, noise_strength=100, area=50000, height=1750) -> None:
        self.height = height
        self.bottom_w = (np.random.random((count, 3)).astype(dtype=float) - 0.5) * 2 \
                        * np.array([area, noise_strength, area], dtype=float) \
                        + np.array([area, 0, 0], dtype=float)
        self.top_w = self.bottom_w + height * np.array([0, 1, 0], dtype=float) + \
                     np.random.uniform(-noise_strength, noise_strength, (count, 3)).astype(dtype=float)

    def __clip__(self) -> np.ndarray:
        zero = np.array([0, 0], dtype=float)
        res = self.camera.intrinsics.res
        mask = np.all(np.logical_and(zero <= self.p_bottom, self.p_bottom < res), axis=1)
        mask = np.logical_and(mask, np.all(np.logical_and(zero <= self.p_top, self.p_top < res), axis=1))
        return mask

    def __addOutliers__(self, ratio=0.2, strength=500):
        count = int(ratio * self.bottom_w.shape[0])
        rand_indices = np.random.choice(self.bottom_w.shape[0],
                                        count,
                                        replace=False)
        self.inliers = ~np.isin(range(self.bottom_w.shape[0]), rand_indices)
        self.bottom_w[~self.inliers] += np.random.uniform(-strength,
                                                          strength,
                                                          (count, 3)).astype(dtype=float)
        self.top_w[~self.inliers] += np.random.uniform(-strength,
                                                       strength,
                                                       (count, 3)).astype(dtype=float)

    def __setUpCamera__(self) -> None:
        self.camera = Camera(extrinsics=Extrinsics(normal=normalize(np.array([0.1, 1, 0.5], dtype=float)),
                                                   distance=20000),
                             intrinsics=Intrinsics(cx=0,
                                                   cy=0,
                                                   fx=1.5,
                                                   fy=1.5,
                                                   res=np.array([800, 600], dtype=float)))

    def setUp(self) -> None:
        self.verbose = False
        self.tolerance = 2e-2
        self.__setUpCamera__()
        self.__setUpScene__()
        self.__addOutliers__(0.1)
        self.p_bottom, self.lambda_b = project(self.bottom_w, self.camera)
        self.p_top, self.lambda_t = project(self.top_w, self.camera)
        self.clip = self.__clip__()
        self.calibrator = LinearCalibrator(self.camera.intrinsics, self.height)

    def test_without_outliers(self) -> None:
        camera = self.calibrator.calibrate(p_bottom=self.p_bottom[self.clip], p_top=self.p_top[self.clip])
        if self.verbose:
            ax = draw_3d_points(self.bottom_w[self.clip & self.inliers], c='green', last=False)
            ax = draw_3d_points(self.top_w[self.clip & self.inliers], c='darkgreen', ax=ax, last=False)
            draw_camera(ax=ax, camera=self.camera)
            draw_3d_points(self.bottom_w[~self.clip], c='black', ax=ax)
            ax = draw_2d_points(self.p_bottom[self.clip & self.inliers], c='lightgreen', last=False)
            draw_2d_points(self.p_top[self.clip & self.inliers], c='darkgreen', ax=ax, res=camera.intrinsics.res)
        self.assertGreater(self.tolerance, np.linalg.norm(np.cross(self.camera.extrinsics.normal, camera.extrinsics.normal)))
        self.assertAlmostEqual(1, self.camera.extrinsics.distance / camera.extrinsics.distance, delta=self.tolerance)

    def test_with_outliers(self) -> None:
        camera = self.calibrator.calibrate(p_bottom=self.p_bottom[self.clip], p_top=self.p_top[self.clip])
        if self.verbose:
            ax = draw_3d_points(self.bottom_w[self.clip & self.inliers], c='green', last=False)
            ax = draw_3d_points(self.top_w[self.clip & self.inliers], c='darkgreen', ax=ax, last=False)
            ax = draw_3d_points(self.bottom_w[self.clip & ~self.inliers], c='red', ax=ax, last=False)
            ax = draw_3d_points(self.top_w[self.clip & ~self.inliers], c='darkred', ax=ax, last=False)
            draw_camera(ax=ax, camera=self.camera)
            draw_3d_points(self.bottom_w[~self.clip], c='black', ax=ax)
            ax = draw_2d_points(self.p_bottom[self.clip & self.inliers], c='lightgreen', last=False)
            ax = draw_2d_points(self.p_top[self.clip & self.inliers], c='darkgreen', ax=ax, last=False)
            ax = draw_2d_points(self.p_bottom[self.clip & ~self.inliers], c='red', ax=ax, last=False)
            draw_2d_points(self.p_top[self.clip & ~self.inliers], c='darkred', ax=ax, res=camera.intrinsics.res)
        self.assertGreater(self.tolerance, np.linalg.norm(np.cross(self.camera.extrinsics.normal, camera.extrinsics.normal)))
        self.assertAlmostEqual(1, self.camera.extrinsics.distance / camera.extrinsics.distance, delta=self.tolerance)

    def test_extensive(self) -> None:
        test_count = 100
        success_count = 0
        for i in range(test_count):
            print(f'test {i + 1} out of {test_count}', end='\r')
            self.setUp()
            camera = self.calibrator.calibrate(p_bottom=self.p_bottom[self.clip], p_top=self.p_top[self.clip])
            if self.tolerance > np.linalg.norm(np.cross(self.camera.extrinsics.normal, camera.extrinsics.normal)) and \
               self.tolerance > abs(1 - self.camera.extrinsics.distance / camera.extrinsics.distance):
                success_count += 1
        print(f'\naccuracy {"{:.0%}".format(success_count / test_count)}', end='\n')
        self.assertGreater(success_count / test_count, 0.9)


class RansacTest(LinearTest):
    def setUp(self) -> None:
        super().setUp()
        self.calibrator = RansacCalibrator(self.camera.intrinsics, self.height)


if __name__ == '__main__':
    unittest.main()
