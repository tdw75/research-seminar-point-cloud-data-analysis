from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data


def plot_cloud(cloud: Union[Data, np.ndarray]):
    if isinstance(cloud, Data):
        cloud = cloud.pos.numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])
    ax.set_axis_off()
    plt.show()
