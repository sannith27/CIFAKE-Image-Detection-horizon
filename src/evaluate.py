"""
Evaluation and visualization for CIFAKE.
"""

import argparse
import json
import os
from typing import Dict, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.data_loader import CIFAKEDataLoader
from src.model import load_trained_model
from src.utils import format_confidence, get_compute_device, get_logger

logger = get_logger(__name__)


def _collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Collect labels and model probabilities from a dataset."""
    y_true = []
    y_pred_probs = []
    compute_device = get_compute_device()
    logger.info("Collecting test predictions on %s.", compute_device)
    for images, labels in dataset:
        with tf.device(compute_device):
            probs = model.predict(images, verbose=0).reshape(-1)
        y_pred_probs.extend(probs.tolist())
        y_true.extend(labels.numpy().reshape(-1).tolist())

    if not y_true:
        raise ValueError("Evaluation dataset is empty.")

    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred_labels = (y_pred_probs >= config.CLASSIFICATION_THRESHOLD).astype(int)
    return y_true, y_pred_probs, y_pred_labels


def _safe_roc_auc(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Return ROC-AUC, or NaN if only one class exists."""
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC-AUC is undefined because y_true contains only one class.")
        return float("nan")
    return float(roc_auc_score(y_true, y_pred_probs))


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    save_plots: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained model on the test dataset."""
    y_true, y_pred_probs, y_pred_labels = _collect_predictions(model, test_dataset)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_labels)),
        "precision": float(precision_score(y_true, y_pred_labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_labels, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred_labels, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_pred_probs),
    }

    for name, value in metrics.items():
        suffix = "" if np.isnan(value) else f" ({format_confidence(value)})"
        logger.info("%s: %.4f%s", name, value, suffix)

    report = classification_report(
        y_true,
        y_pred_labels,
        labels=[0, 1],
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )
    report_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(report)
    logger.info("Classification report saved to %s.", report_path)

    metrics_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)
    logger.info("Evaluation metrics saved to %s.", metrics_path)

    cmatrix = confusion_matrix(y_true, y_pred_labels, labels=[0, 1])
    cmatrix_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "confusion_matrix.csv")
    pd.DataFrame(cmatrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).to_csv(cmatrix_path)
    logger.info("Confusion matrix values saved to %s.", cmatrix_path)

    if save_plots:
        plot_confusion_matrix(y_true, y_pred_labels)
        plot_roc_curve(y_true, y_pred_probs)
        plot_precision_recall_curve(y_true, y_pred_probs)

    return metrics


def plot_training_history(
    history: Optional[tf.keras.callbacks.History] = None,
    history_csv_path: str = config.HISTORY_CSV_PATH,
) -> None:
    """Plot train/validation accuracy and loss curves."""
    if history is not None:
        hist_df = pd.DataFrame(history.history)
    elif os.path.isfile(history_csv_path):
        hist_df = pd.read_csv(history_csv_path)
    else:
        logger.warning("No training history found at %s.", history_csv_path)
        return

    required_columns = {"accuracy", "loss"}
    if not required_columns.issubset(hist_df.columns):
        logger.warning("History is missing required columns: %s", required_columns - set(hist_df.columns))
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist_df["accuracy"], label="Train Accuracy")
    if "val_accuracy" in hist_df.columns:
        axes[0].plot(hist_df["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist_df["loss"], label="Train Loss")
    if "val_loss" in hist_df.columns:
        axes[1].plot(hist_df["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "accuracy_loss_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to %s.", save_path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a confusion matrix heatmap."""
    cmatrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cmatrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s.", save_path)


def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray) -> None:
    """Save an ROC curve if both classes are present."""
    if len(np.unique(y_true)) < 2:
        logger.warning("Skipping ROC curve because only one class is present.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "roc_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s.", save_path)


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_probs: np.ndarray) -> None:
    """Save a Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="teal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_GRAPHS_DIR, "precision_recall_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Precision-Recall curve saved to %s.", save_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAKE model.")
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trained_model = load_trained_model(args.model)
    loader = CIFAKEDataLoader()
    _, _, test_ds = loader.load_datasets(augment_train=False)
    evaluate_model(trained_model, test_ds)
    plot_training_history()
