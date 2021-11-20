import numpy as np
from scipy.optimize import least_squares

from src.detection import BoundingBox
from src.projection.projection import back_project, project
from src.projection.base import Extrinsics, Camera, normalize
from src.projection.calibrator import ProjectionCalibrator


class LeastSquaresCalibrator(ProjectionCalibrator):
    def __back_projected_height__(self, bbox: BoundingBox, parameters: np.ndarray) -> float:
        w, h = self.camera.intrinsics.res.tolist()
        x0, x1, x2 = parameters.tolist()
        xmin, ymin, xmax, ymax = bbox.x, h-bbox.y-bbox.h, bbox.x+bbox.w, h-bbox.y
        p = (0.5-ymin/h)*x1
        f = np.sqrt(1+2*(xmin+xmax-w)/h*np.cos(p)*np.tan(x1/2)**2)
        f = f*x0/np.sin(x2-p)*np.cos(p)*np.tan(x1/2)*(ymax-ymin)/h
        return 2*f

    def __loss__(self, parameters: np.ndarray) -> np.ndarray:
        camera = Camera(intrinsics=self.camera.intrinsics,
                        extrinsics=Extrinsics(distance=parameters[3],
                                              normal=normalize(parameters[:3])))
        points = back_project(self.pixels_b, camera)
        points[:, 1] += self.person_height
        pixels_t_new, scaling_factors = project(points, camera)
        pixel_diff = pixels_t_new - self.pixels_t
        rel_diff = pixel_diff / camera.intrinsics.res
        res = np.linalg.norm(rel_diff, axis=-1)
        d = camera.extrinsics.distance * camera.extrinsics.normal.dot(np.array([0, 0, 1], dtype=float))
        #res = res * (np.abs(scaling_factors)) / d
        med = np.median(res)
        avg = np.average(res)
        dev = np.std(res)
        return res

    @staticmethod
    def __solve_homogeneous_ls__(mtx: np.ndarray) -> np.ndarray:
        eigen_pairs = np.linalg.eig(np.dot(mtx.T, mtx))
        min_index = np.argmin(eigen_pairs[0])
        return eigen_pairs[1][:, min_index]

    def calibrate(self, p_top: np.ndarray, p_bottom: np.ndarray) -> Camera:
        self.__validate_input__(p_top, p_bottom)
        self.pixels_t = p_top
        self.pixels_b = p_bottom
        initial_parameters = np.array([*self.camera.extrinsics.normal,
                                       self.camera.extrinsics.distance], dtype=float)
        solution = least_squares(self.__loss__, initial_parameters)
        self.camera.extrinsics.normal = normalize(solution.x[:3])
        self.camera.extrinsics.distance = solution.x[3]
        return self.camera

