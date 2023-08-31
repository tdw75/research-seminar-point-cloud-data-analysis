from typing import Dict

from torch_geometric.datasets import ModelNet

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader


def init_train_data_loader(
    root: str,
    num_points: int = 1024,
    num_classes: int = 10,
    is_with_affine_transformations: bool = False,
    **kwargs
) -> DataLoader:
    kwargs = {"batch_size": 32, "shuffle": True, "num_workers": 2, **kwargs}
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points)
    if is_with_affine_transformations:
        return init_single(
            root,
            pre_transform,
            T.Compose(
                [
                    transform,
                    T.RandomFlip(0, 0.05),
                    T.RandomFlip(1, 0.05),
                    T.RandomFlip(2, 0.05),
                    T.RandomRotate(180),
                    T.RandomScale((0.5, 1.5)),
                ]
            ),
            num_classes=num_classes,
            is_train=True,
            **kwargs
        )
    else:
        return init_single(
            root,
            pre_transform,
            transform,
            num_classes=num_classes,
            is_train=True,
            **kwargs
        )


def init_test_data_loaders(
    root: str,
    train_loader: DataLoader,
    num_classes: int = 10,
    is_with_affine_transformations: bool = False,
) -> Dict[str, DataLoader]:

    pre_transform = train_loader.dataset.pre_transform
    transform = train_loader.dataset.transform
    kwargs = {
        "batch_size": train_loader.batch_size,
        "shuffle": False,
        "num_workers": train_loader.num_workers,
    }

    loaders = {
        "original": init_single(
            root,
            pre_transform,
            transform,
            num_classes=num_classes,
            is_train=False,
            **kwargs
        )
    }
    if is_with_affine_transformations:
        loaders_trans = add_affine_transformations(
            root, pre_transform, transform, num_classes, is_train=False, **kwargs
        )
        loaders.update(loaders_trans)
    return loaders


def add_affine_transformations(
    root: str,
    pre_transform,
    main_transform,
    num_classes: int = 10,
    is_train: bool = False,
    flip_prob: float = 0.25,
    rotate_deg: int = 180,
    scale_max: float = 2,
    **kwargs
) -> Dict[str, DataLoader]:
    return {
        "flipped": init_single(
            root,
            pre_transform,
            main_transform,
            T.Compose(
                [
                    T.RandomFlip(0, flip_prob),
                    T.RandomFlip(1, flip_prob),
                    T.RandomFlip(2, flip_prob),
                ]
            ),
            num_classes,
            is_train,
            **kwargs
        ),
        "rotated": init_single(
            root,
            pre_transform,
            main_transform,
            T.RandomRotate(rotate_deg),
            num_classes,
            is_train,
            **kwargs
        ),
        "scaled": init_single(
            root,
            pre_transform,
            main_transform,
            T.RandomScale((0, scale_max)),
            num_classes,
            is_train,
            **kwargs
        ),
    }


def init_single(
    root: str,
    pre_transform,
    main_transform,
    affine_transform=None,
    num_classes: int = 10,
    is_train: bool = False,
    **kwargs
) -> DataLoader:
    transforms = (
        T.Compose([main_transform, affine_transform])
        if affine_transform is not None
        else main_transform
    )
    return DataLoader(
        ModelNet(root, str(num_classes), is_train, transforms, pre_transform), **kwargs
    )
