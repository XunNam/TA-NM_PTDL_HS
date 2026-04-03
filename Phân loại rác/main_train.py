from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.dataset import GarbageDataset
from src.evaluate import evaluate_model
from src.model_factory import build_model
from src.train import fit_model
from src.transforms import get_eval_transform, get_train_transform
from src.utils import (
    ensure_dir,
    get_device,
    plot_confusion_matrix,
    plot_training_curves,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a garbage image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/interim/images"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def build_dataloader(
    manifest_path: Path,
    images_root: Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = GarbageDataset(manifest_path=manifest_path, images_root=images_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=str(args.data_dir),
        splits_dir=str(args.splits_dir),
        output_dir=str(args.output_dir),
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        seed=args.seed,
    )

    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    project_root = Path.cwd()
    data_dir = project_root / config.data_dir
    splits_dir = project_root / config.splits_dir
    output_dir = project_root / config.output_dir
    models_dir = ensure_dir(output_dir / "models")
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")

    train_manifest = splits_dir / "train.csv"
    val_manifest = splits_dir / "val.csv"
    test_manifest = splits_dir / "test.csv"
    for manifest in (train_manifest, val_manifest, test_manifest):
        if not manifest.exists():
            raise FileNotFoundError(f"Required manifest not found: {manifest}")

    manifest_df = pd.concat(
        [pd.read_csv(train_manifest), pd.read_csv(val_manifest), pd.read_csv(test_manifest)],
        ignore_index=True,
    )
    class_mapping_df = (
        manifest_df[["label", "class_idx"]]
        .drop_duplicates()
        .sort_values("class_idx")
        .reset_index(drop=True)
    )
    class_names = class_mapping_df["label"].tolist()
    class_to_idx = {
        row["label"]: int(row["class_idx"]) for _, row in class_mapping_df.iterrows()
    }
    num_classes = len(class_names)

    train_transform = get_train_transform(config.img_size, config.mean, config.std)
    eval_transform = get_eval_transform(config.img_size, config.mean, config.std)

    train_loader = build_dataloader(
        manifest_path=train_manifest,
        images_root=data_dir,
        transform=train_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        manifest_path=val_manifest,
        images_root=data_dir,
        transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    test_loader = build_dataloader(
        manifest_path=test_manifest,
        images_root=data_dir,
        transform=eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    sample_inputs, sample_targets = next(iter(train_loader))
    print(
        "Dataloader smoke test:",
        f"batch_shape={tuple(sample_inputs.shape)}",
        f"labels_shape={tuple(sample_targets.shape)}",
    )

    model = build_model(
        model_name=config.model_name,
        num_classes=num_classes,
        pretrained=config.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    checkpoint_path = models_dir / "best_model.pt"
    checkpoint_payload = {
        "model_name": config.model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "img_size": config.img_size,
        "mean": list(config.mean),
        "std": list(config.std),
        "seed": config.seed,
    }
    history, best_info = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(metrics_dir / "history.csv", index=False)
    plot_training_curves(history_df, figures_dir / "training_curves.png")

    test_metrics, report_df, confusion = evaluate_model(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
    )
    report_df.to_csv(metrics_dir / "classification_report.csv", index=False)
    plot_confusion_matrix(confusion, class_names, figures_dir / "confusion_matrix.png")

    metadata = {
        "model_name": config.model_name,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "img_size": config.img_size,
        "mean": list(config.mean),
        "std": list(config.std),
        "best_epoch": best_info["best_epoch"],
        "best_val_accuracy": best_info["best_val_accuracy"],
        "device_used_for_training": str(device),
    }
    save_json(metadata, models_dir / "model_metadata.json")
    save_json(config.to_dict(), models_dir / "training_config.json")
    save_json(
        {
            **test_metrics,
            "best_epoch": best_info["best_epoch"],
            "best_val_accuracy": best_info["best_val_accuracy"],
        },
        metrics_dir / "test_metrics.json",
    )

    print("Training complete.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
