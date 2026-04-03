from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import yaml

from .config import DataPrepConfig, SUPPORTED_IMAGE_EXTENSIONS
from .utils import compute_file_sha1, ensure_dir, save_json, set_seed, slugify


LOGGER = logging.getLogger(__name__)

SPLIT_ALIASES = {
    "train": {"train", "training"},
    "val": {"val", "valid", "validation"},
    "test": {"test", "testing"},
}
ALL_SPLIT_ALIASES = set().union(*SPLIT_ALIASES.values())
GENERIC_DIR_TOKENS = {
    "data",
    "dataset",
    "datasets",
    "image",
    "images",
    "img",
    "imgs",
    "raw",
    "download",
    "downloads",
    "extracted",
    "archive",
    "archives",
    "kaggle",
    "kaggle_source",
}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the garbage dataset.")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_archive_file(path: Path) -> bool:
    suffixes = path.suffixes
    if not suffixes:
        return False
    compound_suffix = "".join(suffixes[-2:]).lower()
    if compound_suffix in {".tar.gz", ".tar.bz2", ".tar.xz"}:
        return True
    return path.suffix.lower() in ARCHIVE_SUFFIXES


def archive_base_name(path: Path) -> str:
    name = path.name
    for suffix in path.suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or path.stem


def detect_split_hint(parts: Iterable[str]) -> str | None:
    for part in parts:
        token = slugify(part)
        for split_name, aliases in SPLIT_ALIASES.items():
            if token in aliases:
                return split_name
    return None


def infer_label(path: Path, root: Path) -> str | None:
    relative_parts = path.relative_to(root).parts[:-1]
    for part in reversed(relative_parts):
        token = slugify(part)
        if token in ALL_SPLIT_ALIASES or token in GENERIC_DIR_TOKENS:
            continue
        return token
    return None


def load_class_names(candidate_roots: list[Path]) -> tuple[list[str] | None, Path | None]:
    for root in candidate_roots:
        yaml_path = root / "data.yaml"
        if not yaml_path.exists():
            continue
        metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = metadata.get("names")
        if isinstance(names, list) and names:
            return [slugify(str(name)) for name in names], yaml_path
    return None, None


def contains_images(root: Path) -> bool:
    return any(path.is_file() and is_image_file(path) for path in root.rglob("*"))


def stage_dataset_source(
    source_path: Path, raw_download_dir: Path, raw_extracted_dir: Path
) -> list[Path]:
    ensure_dir(raw_download_dir)
    ensure_dir(raw_extracted_dir)
    candidate_roots: list[Path] = []

    if source_path.is_dir():
        copied_root = raw_extracted_dir / "kaggle_source"
        shutil.copytree(source_path, copied_root, dirs_exist_ok=True)
        candidate_roots.append(copied_root)

        archives = [path for path in copied_root.rglob("*") if path.is_file() and is_archive_file(path)]
        source_contains_images = contains_images(copied_root)
        for index, archive_path in enumerate(sorted(archives), start=1):
            archive_copy = raw_download_dir / f"{index:03d}_{archive_path.name}"
            shutil.copy2(archive_path, archive_copy)

            extract_dir = raw_extracted_dir / "archives" / f"{index:03d}_{slugify(archive_base_name(archive_path))}"
            ensure_dir(extract_dir)
            try:
                shutil.unpack_archive(str(archive_copy), str(extract_dir))
            except (shutil.ReadError, ValueError) as exc:
                LOGGER.warning("Skipping archive %s: %s", archive_copy, exc)
                continue
            if not source_contains_images:
                candidate_roots.append(extract_dir)
        return candidate_roots

    archive_copy = raw_download_dir / source_path.name
    shutil.copy2(source_path, archive_copy)
    if is_archive_file(archive_copy):
        extract_dir = raw_extracted_dir / slugify(archive_base_name(archive_copy))
        ensure_dir(extract_dir)
        shutil.unpack_archive(str(archive_copy), str(extract_dir))
        candidate_roots.append(extract_dir)
    return candidate_roots


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def build_canonical_filename(source_path: Path, sha1: str) -> str:
    stem = slugify(source_path.stem)
    extension = source_path.suffix.lower() or ".jpg"
    return f"{stem}__{sha1[:12]}{extension}"


def infer_label_from_annotation(
    source_path: Path, class_names: list[str] | None
) -> tuple[str | None, dict]:
    if not class_names or source_path.parent.name != "images":
        return None, {}

    label_path = source_path.parent.parent / "labels" / f"{source_path.stem}.txt"
    if not label_path.exists():
        return None, {}

    class_ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(float(line.split()[0])))

    if not class_ids:
        return None, {}

    counts = Counter(class_ids)
    majority_class_id = counts.most_common(1)[0][0]
    if majority_class_id >= len(class_names):
        return None, {}

    return class_names[majority_class_id], {
        "label_source": "annotation_majority",
        "annotation_num_boxes": len(class_ids),
        "annotation_unique_classes": len(counts),
        "annotation_majority_class_id": majority_class_id,
        "annotation_label_path": str(label_path),
    }


def collect_records(
    candidate_roots: list[Path], interim_images_dir: Path, class_names: list[str] | None
) -> tuple[pd.DataFrame, list[dict]]:
    records: list[dict] = []
    corrupt_files: list[dict] = []
    seen_source_paths: set[str] = set()
    skipped_without_label: list[str] = []

    for root in candidate_roots:
        for source_path in sorted(root.rglob("*")):
            if not source_path.is_file() or not is_image_file(source_path):
                continue

            key = str(source_path.resolve())
            if key in seen_source_paths:
                continue
            seen_source_paths.add(key)

            try:
                validate_image(source_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt_files.append({"source_path": str(source_path), "error": str(exc)})
                continue

            label, annotation_info = infer_label_from_annotation(source_path, class_names)
            if label is None:
                label = infer_label(source_path, root)
                annotation_info = {"label_source": "directory"} if label is not None else {}
            if label is None:
                skipped_without_label.append(str(source_path))
                continue

            sha1 = compute_file_sha1(source_path)
            split_hint = detect_split_hint(source_path.relative_to(root).parts[:-1])
            destination_dir = ensure_dir(interim_images_dir / label)
            destination_path = destination_dir / build_canonical_filename(source_path, sha1)
            if not destination_path.exists():
                shutil.copy2(source_path, destination_path)

            records.append(
                {
                    "source_root": str(root),
                    "source_path": str(source_path),
                    "sha1": sha1,
                    "label": label,
                    "split_hint": split_hint,
                    "relative_path": destination_path.relative_to(interim_images_dir).as_posix(),
                    **annotation_info,
                }
            )

    if not records:
        raise RuntimeError("No valid image files were found after dataset staging.")

    if skipped_without_label:
        LOGGER.warning(
            "Skipped %d image files because no class label could be inferred.",
            len(skipped_without_label),
        )

    return pd.DataFrame(records), corrupt_files


def summarise_group_hints(group: pd.Series) -> str | None:
    hints = sorted({value for value in group.dropna().tolist()})
    if not hints:
        return None
    if len(hints) == 1:
        return hints[0]
    return "conflict"


def build_group_frame(records_df: pd.DataFrame) -> pd.DataFrame:
    return (
        records_df.groupby("sha1")
        .agg(
            label=("label", lambda values: sorted(values)[0]),
            split_hint=("split_hint", summarise_group_hints),
            sample_count=("source_path", "count"),
            unique_labels=("label", "nunique"),
        )
        .reset_index()
    )


def split_groups(
    group_df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    stratify = group_df["label"] if group_df["label"].nunique() > 1 else None
    try:
        left_idx, right_idx = train_test_split(
            group_df.index,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        return group_df.loc[left_idx].copy(), group_df.loc[right_idx].copy(), stratify is not None
    except ValueError:
        left_idx, right_idx = train_test_split(
            group_df.index,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return group_df.loc[left_idx].copy(), group_df.loc[right_idx].copy(), False


def assign_splits(
    records_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    group_df = build_group_frame(records_df)
    strategy = "fresh_split"
    split_notes: list[str] = []

    if (group_df["unique_labels"] > 1).any():
        split_notes.append("Duplicate file hashes with conflicting labels were detected.")

    all_hints_clean = group_df["split_hint"].notna().all() and not (group_df["split_hint"] == "conflict").any()
    hinted_splits = set(group_df["split_hint"].dropna().tolist())

    if all_hints_clean and {"train", "val", "test"}.issubset(hinted_splits):
        group_df["split"] = group_df["split_hint"]
        strategy = "preserved_existing_train_val_test"
        split_notes.append("Used existing train/val/test hints from the dataset.")
    elif all_hints_clean and hinted_splits == {"train", "test"}:
        strategy = "preserved_existing_test_created_val_from_train"
        train_groups = group_df[group_df["split_hint"] == "train"].copy()
        test_groups = group_df[group_df["split_hint"] == "test"].copy()
        val_fraction = val_ratio / (train_ratio + val_ratio)
        train_groups, val_groups, used_stratified = split_groups(
            train_groups, test_size=val_fraction, seed=seed
        )
        split_notes.append(
            "Created validation split from existing train split "
            f"({'stratified' if used_stratified else 'random fallback'})."
        )
        train_groups["split"] = "train"
        val_groups["split"] = "val"
        test_groups["split"] = "test"
        group_df = pd.concat([train_groups, val_groups, test_groups], ignore_index=True)
    else:
        train_val_groups, test_groups, used_test_stratify = split_groups(
            group_df, test_size=test_ratio, seed=seed
        )
        val_fraction = val_ratio / (train_ratio + val_ratio)
        train_groups, val_groups, used_val_stratify = split_groups(
            train_val_groups, test_size=val_fraction, seed=seed
        )
        split_notes.append(
            "Created fresh train/val/test splits "
            f"(test split: {'stratified' if used_test_stratify else 'random fallback'}, "
            f"val split: {'stratified' if used_val_stratify else 'random fallback'})."
        )
        train_groups["split"] = "train"
        val_groups["split"] = "val"
        test_groups["split"] = "test"
        group_df = pd.concat([train_groups, val_groups, test_groups], ignore_index=True)

    class_to_idx = {
        label: index for index, label in enumerate(sorted(records_df["label"].unique()))
    }
    split_map = group_df.set_index("sha1")["split"]
    assigned_df = records_df.copy()
    assigned_df["split"] = assigned_df["sha1"].map(split_map)
    assigned_df["class_idx"] = assigned_df["label"].map(class_to_idx)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_hashes = set(assigned_df.loc[assigned_df["split"] == left, "sha1"])
        right_hashes = set(assigned_df.loc[assigned_df["split"] == right, "sha1"])
        overlap = left_hashes.intersection(right_hashes)
        if overlap:
            raise RuntimeError(f"Detected overlapping file hashes between {left} and {right}.")

    return assigned_df, {"strategy": strategy, "notes": split_notes}


def save_split_manifests(assigned_df: pd.DataFrame, splits_dir: Path) -> None:
    ensure_dir(splits_dir)
    for split_name in ("train", "val", "test"):
        split_df = (
            assigned_df.loc[assigned_df["split"] == split_name, ["relative_path", "label", "class_idx"]]
            .sort_values(["label", "relative_path"])
            .reset_index(drop=True)
        )
        split_df.to_csv(splits_dir / f"{split_name}.csv", index=False)


def save_dataset_summary(
    assigned_df: pd.DataFrame,
    summary_path: Path,
    corrupt_files: list[dict],
) -> None:
    rows: list[dict] = []

    for label, count in assigned_df["label"].value_counts().sort_index().items():
        rows.append({"summary_type": "overall_class_count", "split": "all", "label": label, "count": int(count)})

    for (split, label), count in (
        assigned_df.groupby(["split", "label"]).size().sort_index().items()
    ):
        rows.append({"summary_type": "split_class_count", "split": split, "label": label, "count": int(count)})

    for split, count in assigned_df["split"].value_counts().sort_index().items():
        rows.append({"summary_type": "split_total", "split": split, "label": "all", "count": int(count)})

    rows.append(
        {
            "summary_type": "corrupt_skipped",
            "split": "all",
            "label": "all",
            "count": int(len(corrupt_files)),
        }
    )

    summary_df = pd.DataFrame(rows)
    ensure_dir(summary_path.parent)
    summary_df.to_csv(summary_path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = DataPrepConfig(
        dataset_id=args.dataset_id,
        output_root=str(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    ratio_sum = round(config.train_ratio + config.val_ratio + config.test_ratio, 6)
    if ratio_sum != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    set_seed(config.seed)

    project_root = Path.cwd()
    output_root = project_root / config.output_root
    raw_download_dir = output_root / "raw" / "download"
    raw_extracted_dir = output_root / "raw" / "extracted"
    interim_images_dir = output_root / "interim" / "images"
    splits_dir = output_root / "splits"
    metrics_dir = project_root / "outputs" / "metrics"

    for directory in (raw_download_dir, raw_extracted_dir, interim_images_dir, splits_dir, metrics_dir):
        ensure_dir(directory)

    LOGGER.info("Downloading dataset %s via kagglehub...", config.dataset_id)
    source_path = Path(kagglehub.dataset_download(config.dataset_id))
    LOGGER.info("Dataset download path: %s", source_path)

    candidate_roots = stage_dataset_source(
        source_path=source_path,
        raw_download_dir=raw_download_dir,
        raw_extracted_dir=raw_extracted_dir,
    )
    if not candidate_roots:
        raise RuntimeError("Dataset staging completed without any candidate roots to inspect.")

    class_names, data_yaml_path = load_class_names(candidate_roots)
    records_df, corrupt_files = collect_records(candidate_roots, interim_images_dir, class_names)
    assigned_df, split_info = assign_splits(
        records_df=records_df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    save_split_manifests(assigned_df, splits_dir)
    save_dataset_summary(assigned_df, metrics_dir / "dataset_summary.csv", corrupt_files)

    if corrupt_files:
        pd.DataFrame(corrupt_files).to_csv(metrics_dir / "corrupt_images.csv", index=False)

    dataset_info = {
        "config": config.to_dict(),
        "download_path": str(source_path),
        "candidate_roots": [str(path) for path in candidate_roots],
        "data_yaml_path": str(data_yaml_path) if data_yaml_path is not None else None,
        "num_images": int(len(assigned_df)),
        "num_unique_hashes": int(assigned_df["sha1"].nunique()),
        "class_names": sorted(assigned_df["label"].unique().tolist()),
        "split_counts": {
            split: int(count) for split, count in assigned_df["split"].value_counts().sort_index().items()
        },
        "class_counts": {
            label: int(count) for label, count in assigned_df["label"].value_counts().sort_index().items()
        },
        "split_strategy": split_info,
        "num_corrupt_skipped": int(len(corrupt_files)),
        "num_mixed_annotation_images": int(
            (assigned_df.get("annotation_unique_classes", pd.Series(dtype=int)).fillna(1) > 1).sum()
        ),
    }
    save_json(dataset_info, metrics_dir / "dataset_info.json")
    save_json({"data_prep_config": asdict(config)}, metrics_dir / "data_prep_config.json")

    print("Prepared dataset successfully.")
    print(f"Images copied to: {interim_images_dir}")
    print(f"Split manifests saved to: {splits_dir}")
    print(f"Dataset summary saved to: {metrics_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
