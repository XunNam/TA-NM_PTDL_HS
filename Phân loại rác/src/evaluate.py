from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=int)
    y_pred = (
        np.concatenate(all_predictions) if all_predictions else np.array([], dtype=int)
    )
    return y_true, y_pred


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_true, y_pred = collect_predictions(model=model, dataloader=dataloader, device=device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "num_samples": int(len(y_true)),
    }
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index(names="label")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return metrics, report_df, matrix
