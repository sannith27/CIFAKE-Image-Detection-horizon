"""
Grad-CAM explainability for the CIFAKE CNN.
"""

import argparse
import os
from typing import Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import colormaps

from src import config
from src.model import load_trained_model
from src.preprocessing import preprocess_single_image
from src.utils import format_confidence, get_compute_device, get_logger, get_predicted_label

logger = get_logger(__name__)
_GRADCAM_MODEL_CACHE: dict[tuple[int, str], tf.keras.Model] = {}


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the configured or last Conv2D layer."""
    layer_names = [layer.name for layer in model.layers]
    if config.GRADCAM_LAST_CONV_LAYER in layer_names:
        return config.GRADCAM_LAST_CONV_LAYER

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            logger.warning(
                "Configured Grad-CAM layer %s not found. Using %s.",
                config.GRADCAM_LAST_CONV_LAYER,
                layer.name,
            )
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def _get_gradcam_model(model: tf.keras.Model, last_conv_layer_name: str) -> tf.keras.Model:
    """Return a cached model that exposes the selected conv layer and prediction."""
    cache_key = (id(model), last_conv_layer_name)
    grad_model = _GRADCAM_MODEL_CACHE.get(cache_key)
    if grad_model is None:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        _GRADCAM_MODEL_CACHE[cache_key] = grad_model
    return grad_model


def generate_gradcam(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap.

    class_index uses CIFAKE labels: 0 = FAKE, 1 = REAL. If omitted, the
    predicted class is explained. For sigmoid binary models, REAL uses p
    and FAKE uses 1 - p as the class score.
    """
    if image_tensor.shape.rank != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape (1, height, width, channels).")

    last_conv_layer_name = last_conv_layer_name or _find_last_conv_layer(model)
    grad_model = _get_gradcam_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        real_score = predictions[:, 0]
        if class_index is None:
            class_index = 1 if float(real_score[0]) >= config.CLASSIFICATION_THRESHOLD else 0
        if class_index == 1:
            loss = real_score
        elif class_index == 0:
            loss = 1.0 - real_score
        else:
            raise ValueError("class_index must be 0 (FAKE), 1 (REAL), or None.")

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradient computation returned None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.squeeze(conv_outputs @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) <= 0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / max_value).numpy()


def overlay_heatmap_on_image(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a heatmap on an RGB image."""
    if original_image is None or original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("original_image must be an RGB array with shape (H, W, 3).")
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    height, width = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (width, height))
    colored_heatmap = colormaps[colormap](heatmap_resized)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)
    original_uint8 = np.clip(original_image, 0, 255).astype(np.uint8)
    return cv2.addWeighted(original_uint8, 1 - alpha, colored_heatmap, alpha, 0)


def save_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    alpha: float = config.GRADCAM_ALPHA,
) -> str:
    """Save a Grad-CAM overlay image."""
    overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=alpha)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f"OpenCV could not write Grad-CAM image to {save_path}.")
    logger.info("Grad-CAM overlay saved to %s.", save_path)
    return save_path


def explain_image(
    model: tf.keras.Model,
    image_path: str,
    output_dir: str = config.OUTPUT_GRADCAM_DIR,
    real_probability: Optional[float] = None,
) -> Tuple[str, str, float]:
    """Save a Grad-CAM overlay for one image."""
    input_shape = model.input_shape[1:3]
    preprocessed = preprocess_single_image(image_path, image_size=tuple(input_shape))
    compute_device = get_compute_device()
    if real_probability is None:
        with tf.device(compute_device):
            real_probability = float(model(preprocessed, training=False)[0][0].numpy())

    predicted_label = get_predicted_label(real_probability)
    class_index = config.CLASS_INDICES[predicted_label]
    with tf.device(compute_device):
        heatmap = generate_gradcam(model, preprocessed, class_index=class_index)

    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_gradcam_{predicted_label}.png")
    save_gradcam_overlay(original_rgb, heatmap, save_path)

    confidence = real_probability if predicted_label == "REAL" else 1.0 - real_probability
    logger.info(
        "Grad-CAM complete for %s: %s (%s).",
        image_path,
        predicted_label,
        format_confidence(confidence),
    )
    return predicted_label, save_path, confidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM explanation.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trained_model = load_trained_model(args.model)
    label, path, score = explain_image(trained_model, args.image)
    print(f"Prediction: {label} ({format_confidence(score)}) | Grad-CAM saved to: {path}")
