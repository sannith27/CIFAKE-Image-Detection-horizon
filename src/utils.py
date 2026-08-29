"""
Shared utilities for the CIFAKE project.
"""

import logging
import os
import random
import sys
import time
from functools import wraps
from typing import Optional

import numpy as np

from src import config

_LOGGING_CONFIGURED = False


def _configure_root_logging() -> None:
    """Configure console and file logging once."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(config.LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as exc:
            root_logger.warning("File logging disabled: %s", exc)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    _configure_root_logging()
    return logging.getLogger(name)


logger = get_logger(__name__)


def import_tensorflow():
    """Import TensorFlow with a clear installation error."""
    try:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf

        return tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is not installed. Run: pip install -r requirements.txt"
        ) from exc


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    seed = config.RANDOM_SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    tf = import_tensorflow()
    tf.random.set_seed(seed)
    logger.info("Random seed set to %d.", seed)


def check_gpu() -> bool:
    """Log TensorFlow GPU availability and enable memory growth when possible."""
    try:
        tf = import_tensorflow()
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logger.warning("No GPU detected. Training will run on CPU.")
            return False

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logger.warning("Could not set memory growth for %s: %s", gpu, exc)

        logger.info("Detected %d GPU(s): %s", len(gpus), gpus)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("GPU check failed: %s", exc)
        return False


def get_compute_device() -> str:
    """Return the configured TensorFlow device for inference and evaluation.

    Set ``CIFAKE_DEVICE=cpu`` to force CPU execution. By default, GPU is
    preferred when TensorFlow exposes one, with a safe CPU fallback.
    """
    tf = import_tensorflow()
    requested_device = os.environ.get("CIFAKE_DEVICE", "auto").strip().lower()

    if requested_device not in {"auto", "gpu", "cpu"}:
        raise ValueError("CIFAKE_DEVICE must be one of: auto, gpu, cpu")

    if requested_device == "cpu":
        logger.info("Using TensorFlow device /CPU:0 (forced by CIFAKE_DEVICE).")
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                logger.warning("Could not set memory growth for %s: %s", gpu, exc)
        logger.info("Using TensorFlow device /GPU:0 for inference/evaluation.")
        return "/GPU:0"

    if requested_device == "gpu":
        raise RuntimeError(
            "CIFAKE_DEVICE=gpu was requested, but TensorFlow did not detect a GPU."
        )

    logger.warning("TensorFlow did not detect a GPU; using /CPU:0.")
    return "/CPU:0"


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def validate_dataset_structure(dataset_dir: str) -> bool:
    """Validate that dataset_dir contains FAKE and REAL subdirectories."""
    if not os.path.isdir(dataset_dir):
        logger.error("Dataset directory does not exist: %s", dataset_dir)
        return False

    missing = []
    for class_name in config.CLASS_NAMES:
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            missing.append(class_path)

    if missing:
        logger.error("Missing class directories: %s", ", ".join(missing))
        return False

    return True


def count_images_per_class(dataset_dir: str) -> dict:
    """Count valid image files per class."""
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    counts = {}
    actual_dirs = {}
    if os.path.isdir(dataset_dir):
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                actual_dirs[item.lower()] = item_path

    for class_name in config.CLASS_NAMES:
        class_path = actual_dirs.get(class_name.lower(), os.path.join(dataset_dir, class_name))
        if not os.path.isdir(class_path):
            counts[class_name] = 0
            continue
        counts[class_name] = sum(
            1
            for file_name in os.listdir(class_path)
            if file_name.lower().endswith(valid_extensions)
        )
    return counts


def validate_model_file(model_path: str) -> None:
    """Raise a clear error if a saved model is missing."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train first with "
            "`python main.py train --epochs 50` or pass --model to an existing .keras file."
        )


def timeit(func):
    """Log execution time for a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.info("Function '%s' completed in %.2f seconds.", func.__name__, time.time() - start_time)
        return result

    return wrapper


def format_confidence(score: float) -> str:
    """Format a probability-like score as a percentage."""
    return f"{float(score) * 100:.2f}%"


def get_predicted_label(score: float, threshold: Optional[float] = None) -> str:
    """Convert a sigmoid probability of REAL into FAKE/REAL."""
    threshold = config.CLASSIFICATION_THRESHOLD if threshold is None else threshold
    return config.CLASS_NAMES[1] if float(score) >= threshold else config.CLASS_NAMES[0]
