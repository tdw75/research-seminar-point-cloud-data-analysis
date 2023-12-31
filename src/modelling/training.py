import time

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader


def train_step(
    model: torch.nn.Module, data_loader: DataLoader, optimiser, device: torch.device
) -> float:
    model.train()
    correct = 0

    for data in data_loader:

        # forward step
        data = data.to(device)
        optimiser.zero_grad()
        output = model(data)

        # calculate accuracy
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()

        # backward step
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimiser.step()

    return correct / len(data_loader.dataset)


def val_step(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    optimiser=None,
    n_epochs: int = 50,
    device: torch.device = None,
):
    optimiser = optimiser or torch.optim.Adam(model.parameters(), lr=0.001)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_acc = train_step(model, train_loader, optimiser, device)
        val_acc = val_step(model, val_loader, device)
        end = time.time()
        print(
            f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Run Time: {end-start:.1f} seconds"
        )

    return model
