from __future__ import annotations

from torchvision import transforms


def get_train_transform(
    img_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]
):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_eval_transform(
    img_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]
):
    resize_size = max(int(img_size * 1.14), img_size + 8)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
