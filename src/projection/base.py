from dataclasses import dataclass
from typing import Tuple

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def homogeneous(vectors: np.ndarray) -> np.ndarray:
    assert vectors.ndim in [1, 2]
    if vectors.ndim == 1:
        return np.concatenate([vectors, 1], dtype=vectors.dtype)
    elif vectors.ndim == 2:
        return np.c_[vectors, np.ones(vectors.shape[0], dtype=vectors.dtype)]


def homogeneous_inv(vectors: np.ndarray) -> np.ndarray:
    assert vectors.ndim in [1, 2]
    if vectors.ndim == 1:
        return vectors[:-1] / vectors[-1]
    elif vectors.ndim == 2:
        return vectors[:, :-1] / vectors[:, -1][:, None]


@dataclass
class Extrinsics:
    normal: np.ndarray = normalize(np.array([0.1, -1, 1], dtype=float))
    distance: float = 20000     # in mms

    def cam(self) -> Tuple[np.ndarray, np.ndarray]:
        inv_base, inv_null = self.cam_inv()
        return np.linalg.inv(inv_base), -1*inv_null

    def cam_inv(self) -> Tuple[np.ndarray, np.ndarray]:
        up = normalize(self.normal)
        forward = np.array([0, 0, -1], dtype=float)
        right = normalize(np.cross(forward, up))
        forward = normalize(np.cross(up, right))
        origo = -self.distance * up
        return np.c_[forward, up, right], origo


@dataclass
class Intrinsics:
    res: np.ndarray = np.array([800, 600], dtype=float)
    fx: float = 2.5
    fy: float = 2.5
    cx: float = 0
    cy: float = 0

    def proj(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=float)

    def proj_inv(self) -> np.ndarray:
        return np.linalg.inv(self.proj())


@dataclass
class Camera:
    extrinsics: Extrinsics = Extrinsics()
    intrinsics: Intrinsics = Intrinsics()
