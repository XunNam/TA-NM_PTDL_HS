from __future__ import annotations

from dataclasses import asdict, dataclass, field


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class DataPrepConfig:
    dataset_id: str
    output_root: str = "data"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    supported_extensions: list[str] = field(
        default_factory=lambda: sorted(SUPPORTED_IMAGE_EXTENSIONS)
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    data_dir: str = "data/interim/images"
    splits_dir: str = "data/splits"
    output_dir: str = "outputs"
    model_name: str = "resnet18"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 4
    pretrained: bool = True
    seed: int = 42
    top_k: int = 5
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mean"] = list(self.mean)
        data["std"] = list(self.std)
        return data
