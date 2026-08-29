---
title: CIFAKE Image Detection
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# CIFAKE: Real vs AI-Generated Image Detection

This project trains a TensorFlow/Keras CNN to classify CIFAKE images as:

```text
FAKE = 0
REAL = 1
```

It also supports evaluation metrics, prediction on new images, and Grad-CAM explanations.

## Folder Structure

```text
project/
  dataset/
    train/
      FAKE/
      REAL/
    test/
      FAKE/
      REAL/
  models/
  notebooks/
  src/
    config.py
    utils.py
    data_loader.py
    preprocessing.py
    model.py
    train.py
    evaluate.py
    predict.py
    gradcam.py
  outputs/
    models/
    graphs/
    predictions/
    gradcam/
    logs/
  requirements.txt
  README.md
  main.py
```

The trained final model is saved to:

```text
outputs/models/cifake_final_model.keras
```

## Installation

From this project folder:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This project is compatible with Python 3.13 and TensorFlow 2.21. On native Windows, TensorFlow 2.21 runs on CPU by default. Use Google Colab or WSL2 for GPU acceleration.

For this NVIDIA GPU machine, use WSL2 Ubuntu and install `requirements-gpu-wsl.txt`.
The CUDA runtime libraries must be included in `LD_LIBRARY_PATH` before running
the commands below.

## Dataset Setup

Download CIFAKE from Kaggle:

https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

After extraction, the folders must look exactly like this:

```text
CIFAKE_Project (2)/project/
  dataset/
    train/
      FAKE/
        image_1.jpg
        ...
      REAL/
        image_1.jpg
        ...
    test/
      FAKE/
        image_1.jpg
        ...
      REAL/
        image_1.jpg
        ...
```

Do not place an extra nested folder between `dataset` and `train`/`test`.

## Commands

Run these from:

```powershell
cd "C:\Users\BHAVYA REDDY\Downloads\CIFAKE_Project (2)\project"
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train:

```powershell
python main.py train --epochs 50
```

For a practical local CPU run using a representative sample of the dataset,
set optional batch limits. A value of `0` keeps the default full-dataset behavior:

```powershell
$env:CIFAKE_MAX_TRAIN_BATCHES="32"
$env:CIFAKE_MAX_VAL_BATCHES="8"
$env:CIFAKE_MAX_TEST_BATCHES="16"
python main.py train --epochs 3 --device auto
```

Evaluate:

```powershell
python main.py evaluate

# Require GPU execution for evaluation (fails clearly if TensorFlow cannot see one)
python main.py evaluate --device gpu

# Force CPU execution when needed
python main.py evaluate --device cpu
```

WSL2 GPU example:

```bash
export CIFAKE_BATCH_SIZE=64
export LD_LIBRARY_PATH=/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/curand/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/$USER/cifake-gpu-venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
python main.py train --epochs 1 --device gpu
python main.py evaluate --device gpu
```

Predict one image:

```powershell
python main.py predict --image "C:\path\to\image.jpg"
```

Predict a folder:

```powershell
python main.py predict --dir "C:\path\to\images"
```

Generate Grad-CAM:

```powershell
python main.py gradcam --image "C:\path\to\image.jpg"
```

## Outputs

Training writes:

```text
outputs/models/best_model.keras
outputs/models/cifake_final_model.keras
outputs/graphs/training_history.csv
outputs/graphs/accuracy_loss_curves.png
```

Evaluation writes:

```text
outputs/graphs/classification_report.txt
outputs/graphs/confusion_matrix.png
outputs/graphs/roc_curve.png
outputs/graphs/precision_recall_curve.png
```

Batch prediction writes:

```text
outputs/predictions/batch_predictions.csv
```

The Flask web app accepts user-supplied BMP, GIF, JPG, JPEG, PNG, and WEBP
images up to 10 MB. Start it with:

```powershell
python app.py
```

Grad-CAM writes:

```text
outputs/gradcam/
```

## Notes

- The project uses cross-platform paths through `os.path`.
- The loader validates the dataset before training.
- Validation data is split automatically from `dataset/train` using `VALIDATION_SPLIT` in `src/config.py`.
- Grad-CAM explains the predicted class. For a FAKE prediction it explains `1 - P(REAL)`; for REAL it explains `P(REAL)`.
