from torch_geometric.datasets import ModelNet


def get_modelnet(root: str):
    train = ModelNet(root, name="10", train=True)
    return train
