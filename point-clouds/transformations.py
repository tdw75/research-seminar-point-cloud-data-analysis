from typing import Tuple

import numpy as np


def translate_cloud(
    cloud: np.ndarray, shift_vector: Tuple[float, float, float] = None
) -> np.ndarray:
    if shift_vector is None:
        shift_vector = 0.5 * np.random.randn(1, 3)  # random translation
    else:
        shift_vector = np.array(shift_vector)
    return cloud + shift_vector


def translate_cloud_random(cloud: np.ndarray) -> np.ndarray:
    return translate_cloud(cloud, 0.5 * np.random.randn(1, 3))


def centre_cloud(cloud: np.ndarray) -> np.ndarray:
    return translate_cloud(cloud, -cloud.mean(axis=0))


def flip_cloud(cloud: np.ndarray, axes: Tuple[bool, bool, bool]) -> np.ndarray:
    signs = -np.sign(np.array(axes) - 0.5)
    cloud_centred = centre_cloud(cloud)
    centre_point = cloud.mean(axis=0)
    cloud_flipped = signs * cloud_centred
    return cloud_flipped + centre_point


def flip_cloud_random(cloud: np.ndarray):
    return flip_cloud(cloud, tuple(np.random.binomial(0, 0.5, 3)))


def scale_cloud(cloud: np.ndarray, scale_degree: float) -> np.ndarray:
    return cloud * scale_degree


def scale_cloud_random(cloud: np.ndarray) -> np.ndarray:
    return scale_cloud(cloud, np.random.rand() * 2)
