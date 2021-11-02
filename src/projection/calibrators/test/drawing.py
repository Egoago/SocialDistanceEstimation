from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from src.projection.base import homogeneous_inv, normalize, Camera


def draw_2d_points(points: np.ndarray, c='b', last=True, ax=None, res=None) -> Any:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c=c)
    if points.ndim == 2:
        ax.scatter(points[:, 0], points[:, 1], c=c)
    elif points.ndim == 1:
        ax.scatter(points[0], points[1], c=c)
    if last:
        if res is not None:
            plt.xlim([0, res[0]])
            plt.ylim([0, res[1]])
        plt.show()
    return ax


def draw_camera(camera: Camera, ax: Any) -> None:
    origo = np.array([0, 0, 0, 1], dtype=float)
    length = 1000
    look_at = np.array([0, 0, -1, 0], dtype=float)
    up = np.array([0, 1, 0, 0], dtype=float)
    right = np.array([1, 0, 0, 0], dtype=float)
    C_inv = camera.extrinsics.cam_inv()
    C = camera.extrinsics.cam()
    origo = homogeneous_inv(C_inv.dot(origo.T))
    look_at = length * normalize(look_at.dot(C)[:-1])
    up = length * normalize(up.dot(C)[:-1])
    right = length * normalize(right.dot(C)[:-1])
    ax.scatter(origo[0], origo[2], origo[1], s=50, c='y')
    ax.plot([0, origo[0]], [0, origo[2]], [0, origo[1]], c='black')
    ax.plot([origo[0], origo[0] + look_at[0]],
            [origo[2], origo[2] + look_at[2]],
            [origo[1], origo[1] + look_at[1]], c='r')
    ax.plot([origo[0], origo[0] + up[0]],
            [origo[2], origo[2] + up[2]],
            [origo[1], origo[1] + up[1]], c='g')
    ax.plot([origo[0], origo[0] + right[0]],
            [origo[2], origo[2] + right[2]],
            [origo[1], origo[1] + right[1]], c='b')


def draw_3d_points(points: np.ndarray, c='b', last=True, ax=None) -> Any:
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    if points.ndim == 2:
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=c)
    elif points.ndim == 1:
        ax.scatter(points[0], points[2], points[1], c=c)
    if last:
        ax.scatter(0, 0, 0, c='r', s=50)  # origo
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.invert_yaxis()
        plt.show()
    return ax
