from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .model_factory import build_model
from .transforms import get_eval_transform
from .utils import get_device, load_json


def load_artifacts(
    model_path: Path,
    metadata_path: Path,
    device: torch.device | None = None,
):
    device = device or get_device()
    metadata = load_json(Path(metadata_path))
    checkpoint = torch.load(Path(model_path), map_location=device)

    model = build_model(
        model_name=metadata["model_name"],
        num_classes=len(metadata["class_names"]),
        pretrained=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata, device


def predict_image(
    image_path: Path,
    model_path: Path,
    metadata_path: Path,
    top_k: int = 5,
    device: torch.device | None = None,
) -> dict:
    model, metadata, device = load_artifacts(
        model_path=model_path, metadata_path=metadata_path, device=device
    )
    transform = get_eval_transform(
        img_size=int(metadata["img_size"]),
        mean=tuple(metadata["mean"]),
        std=tuple(metadata["std"]),
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)

    top_k = min(top_k, len(metadata["class_names"]))
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    top_predictions = [
        {
            "label": metadata["class_names"][index],
            "probability": float(prob),
        }
        for prob, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
    ]
    predicted_index = int(top_indices[0].item())
    return {
        "predicted_label": metadata["class_names"][predicted_index],
        "confidence": float(top_probs[0].item()),
        "top_k": top_predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        top_k=args.top_k,
    )
    print(f"Predicted class: {prediction['predicted_label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    for item in prediction["top_k"]:
        print(f"{item['label']}: {item['probability']:.4f}")


if __name__ == "__main__":
    main()
