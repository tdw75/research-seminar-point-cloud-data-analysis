import os
from typing import Tuple

import numpy as np
import trimesh


def parse_dataset(
    directory: str, sample_size: int = 2048, dataset_name: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    labels = []

    unneeded_files = [".DS_Store", "README.txt"]
    label_set = filter_list(sorted(os.listdir(f"{directory}/raw")), unneeded_files)

    for label in label_set:

        path = f"{directory}/raw/{label}/{dataset_name}"
        files = filter_list(os.listdir(path), unneeded_files)

        for f in files:
            points.append(mesh_to_point_cloud(f"{path}/{f}", sample_size))
            labels.append(label)

    return np.array(points), np.array(labels)


def mesh_to_point_cloud(file_path: str, sample_size: int):
    return trimesh.load(file_path).sample(sample_size)


def filter_list(lst: list, unneeded: list) -> list:
    return [x for x in lst if x not in unneeded]
