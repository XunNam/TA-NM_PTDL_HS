from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import ensure_dir


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += inputs.size(0)

    average_loss = total_loss / max(total_samples, 1)
    average_accuracy = total_correct / max(total_samples, 1)
    return average_loss, average_accuracy


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
    checkpoint_payload: dict,
) -> tuple[list[dict], dict]:
    ensure_dir(checkpoint_path.parent)

    history: list[dict] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())
            checkpoint = {
                **checkpoint_payload,
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "model_state_dict": best_state_dict,
            }
            torch.save(checkpoint, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state_dict)
    return history, {"best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy}
