import torch

from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


"""
Code from PointNet++ tutorial in the torch-geometric repository
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
"""


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)  # sampling
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )  # grouping
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__()
        kwargs = {
            "sa1_ratio": 0.5,
            "sa1_radius": 0.2,
            "sa2_ratio": 0.25,
            "sa2_radius": 0.4,
            "dropout": 0.1,
            **kwargs,
        }

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            kwargs["sa1_ratio"], kwargs["sa1_radius"], MLP([3, 64, 64, 128])
        )
        self.sa2_module = SAModule(
            kwargs["sa2_ratio"], kwargs["sa2_radius"], MLP([128 + 3, 128, 128, 256])
        )
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP(
            [1024, 512, 256, num_classes], dropout=kwargs["dropout"], norm=None
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)
