from typing import Dict

from torch_geometric.datasets import ModelNet

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader


def init_train_data_loader(root: str, num_points: int = 1024, **kwargs) -> DataLoader:
    kwargs = {"batch_size": 32, "shuffle": True, "num_workers": 6, **kwargs}
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points)
    train_dataset = ModelNet(root, "10", True, transform, pre_transform)
    return DataLoader(train_dataset, **kwargs)


def init_test_data_loaders(
    root: str, train_loader: DataLoader, is_affine_transformations: bool = False
) -> Dict[str, DataLoader]:

    pre_transform = train_loader.dataset.pre_transform
    transform = train_loader.dataset.transform
    kwargs = {
        "batch_size": train_loader.batch_size,
        "shuffle": False,
        "num_workers": train_loader.num_workers,
    }

    def init_single(affine_transform=None) -> DataLoader:
        transforms = (
            T.Compose([transform, affine_transform])
            if affine_transform is not None
            else transform
        )
        return DataLoader(
            ModelNet(root, "10", False, transforms, pre_transform), **kwargs
        )

    loaders = {"original": init_single()}
    if is_affine_transformations:
        loaders_trans = {
            "flipped": init_single(
                T.Compose(
                    [T.RandomFlip(0, 0.5), T.RandomFlip(1, 0.5), T.RandomFlip(2, 0.5)]
                )
            ),
            "rotated": init_single(T.RandomRotate(180)),
            "scaled": init_single(T.RandomScale((0, 0.5))),
        }
        loaders.update(loaders_trans)
    return loaders
