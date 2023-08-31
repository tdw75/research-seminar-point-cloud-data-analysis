import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch_geometric.loader import DataLoader

LABELS = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]


def evaluate_model(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
):
    y_pred = np.array([])
    y_true = np.array([])

    model.eval()
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]

        y_pred = np.concatenate([y_pred, pred.numpy()])
        y_true = np.concatenate([y_true, data.y.numpy()])

    return y_true, y_pred


def plot_confusion_matrix(true: np.ndarray, pred: np.ndarray):
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap="Blues", xticks_rotation=90)


def get_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return (true == pred).sum() / len(true)


def get_accuracy_by_class(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    results = pd.DataFrame({"true": true, "pred": pred})
    results["correct"] = results["true"] == results["pred"]
    grouped = (
        results.astype(int)
        .groupby(by="true", as_index=False)
        .agg({"correct": "sum", "true": "count"})
    )
    return grouped["correct"] / grouped["true"]
