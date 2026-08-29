"""
Single-image and batch prediction for CIFAKE.
"""

import argparse
import glob
import os
from typing import Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd
import tensorflow as tf

from src import config
from src.model import load_trained_model
from src.preprocessing import preprocess_single_image
from src.utils import format_confidence, get_compute_device, get_logger, get_predicted_label

logger = get_logger(__name__)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


def predict_single_image(model: tf.keras.Model, image_path: str) -> Dict:
    """Predict one image and return label plus confidence details."""
    input_shape = model.input_shape[1:3]
    preprocessed = preprocess_single_image(image_path, image_size=tuple(input_shape))
    with tf.device(get_compute_device()):
        real_probability = float(model(preprocessed, training=False)[0][0].numpy())
    label = get_predicted_label(real_probability)
    predicted_confidence = real_probability if label == "REAL" else 1.0 - real_probability

    result = {
        "image_path": image_path,
        "predicted_label": label,
        "predicted_confidence": round(predicted_confidence, 6),
        "predicted_confidence_percent": format_confidence(predicted_confidence),
        "real_probability": round(real_probability, 6),
        "fake_probability": round(1.0 - real_probability, 6),
    }
    logger.info(
        "Prediction for %s: %s (%s)",
        image_path,
        label,
        result["predicted_confidence_percent"],
    )
    return result


def predict_batch(
    model: tf.keras.Model,
    directory: str,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Predict all valid images in a directory non-recursively."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_paths: List[str] = sorted(
        path
        for path in glob.glob(os.path.join(directory, "*"))
        if path.lower().endswith(VALID_EXTENSIONS)
    )
    if not image_paths:
        raise ValueError(f"No valid images found in directory: {directory}")

    results = []
    for image_path in image_paths:
        try:
            results.append(predict_single_image(model, image_path))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.warning("Skipping %s: %s", image_path, exc)

    if not results:
        raise RuntimeError("No images could be predicted successfully.")

    results_df = pd.DataFrame(results)
    if save_csv:
        csv_path = os.path.join(config.OUTPUT_PREDICTIONS_DIR, "batch_predictions.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info("Batch predictions saved to %s.", csv_path)
    return results_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict REAL/FAKE for image(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--dir", type=str)
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trained_model = load_trained_model(args.model)
    if args.image:
        print(predict_single_image(trained_model, args.image))
    else:
        print(predict_batch(trained_model, args.dir))
