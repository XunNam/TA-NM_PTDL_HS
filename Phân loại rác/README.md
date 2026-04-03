# Garbage Image Classification

University-friendly PyTorch image classification project for the Kaggle dataset `hutinguynhunh/garbage-classification-v2`.

## Project Purpose

This project downloads the dataset with `kagglehub`, inspects and reorganizes the files into a clean local layout, creates reproducible split manifests, trains a transfer-learning classifier, evaluates it, and provides a Jupyter notebook for image-based inference demos.

## Dataset Source

- Kaggle dataset: `hutinguynhunh/garbage-classification-v2`
- Download mechanism used by the project:

```python
import kagglehub

path = kagglehub.dataset_download("hutinguynhunh/garbage-classification-v2")
print(path)
```

## Folder Structure

```text
Garbage/
├─ README.md
├─ .gitignore
├─ requirements_snapshot.txt
├─ data/
│  ├─ raw/
│  │  ├─ download/
│  │  └─ extracted/
│  ├─ interim/
│  │  └─ images/
│  └─ splits/
├─ notebooks/
│  └─ garbage_inference.ipynb
├─ outputs/
│  ├─ models/
│  ├─ metrics/
│  └─ figures/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ dataset.py
│  ├─ evaluate.py
│  ├─ inference.py
│  ├─ model_factory.py
│  ├─ prepare_data.py
│  ├─ train.py
│  ├─ transforms.py
│  └─ utils.py
└─ main_train.py
```

## Environment Activation

```bash
cd /home/buitheanh/Documents/PTDLHS/Garbage
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PTDLHS
```

## Prepare Data

```bash
python -m src.prepare_data \
  --dataset-id hutinguynhunh/garbage-classification-v2 \
  --output-root data \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Outputs:

- Canonical images: `data/interim/images/`
- Split manifests: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`
- Dataset summaries: `outputs/metrics/dataset_summary.csv`, `outputs/metrics/dataset_info.json`

## Train the Model

```bash
python main_train.py \
  --data-dir data/interim/images \
  --splits-dir data/splits \
  --output-dir outputs \
  --model-name resnet18 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --num-workers 4 \
  --patience 5 \
  --img-size 224 \
  --seed 42
```

Training artifacts:

- Best checkpoint: `outputs/models/best_model.pt`
- Model metadata: `outputs/models/model_metadata.json`
- Training config: `outputs/models/training_config.json`
- Training history: `outputs/metrics/history.csv`
- Test metrics: `outputs/metrics/test_metrics.json`
- Classification report: `outputs/metrics/classification_report.csv`
- Figures: `outputs/figures/confusion_matrix.png`, `outputs/figures/training_curves.png`

## Inference Notebook

Open Jupyter in the same conda environment:

```bash
jupyter lab
```

Then open `notebooks/garbage_inference.ipynb`.

The notebook:

- loads the saved model and metadata;
- lets you set an `IMAGE_PATH` in a cell;
- displays the input image;
- predicts the label;
- shows confidence and top-k probabilities.

## Notes

- CUDA is used automatically when available; otherwise the code falls back to CPU.
- Random seeds are set for reproducibility.
- Existing train/val/test folders are preserved when clean split hints are available; otherwise reproducible splits are created from the discovered dataset.
