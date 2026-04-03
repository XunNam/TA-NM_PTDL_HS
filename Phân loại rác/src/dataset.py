from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GarbageDataset(Dataset):
    def __init__(self, manifest_path: Path, images_root: Path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.images_root = Path(images_root)
        self.transform = transform
        self.data = pd.read_csv(self.manifest_path)
        required_columns = {"relative_path", "label", "class_idx"}
        missing = required_columns.difference(self.data.columns)
        if missing:
            raise ValueError(
                f"Manifest {self.manifest_path} is missing columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_path = self.images_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["class_idx"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
