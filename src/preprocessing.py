"""
Preprocessing and data augmentation for CIFAKE.
"""

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras import layers

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def get_augmentation_layer() -> tf.keras.Sequential:
    """Create the training-only augmentation pipeline."""
    aug_cfg = config.AUGMENTATION_CONFIG
    return tf.keras.Sequential(
        [
            layers.RandomFlip(aug_cfg["random_flip"]),
            layers.RandomRotation(aug_cfg["random_rotation"]),
            layers.RandomZoom(aug_cfg["random_zoom"]),
            layers.RandomTranslation(
                height_factor=aug_cfg["random_translation"],
                width_factor=aug_cfg["random_translation"],
            ),
            layers.RandomContrast(aug_cfg["random_contrast"]),
            layers.RandomBrightness(
                factor=aug_cfg["random_brightness"],
                value_range=(0.0, 1.0),
            ),
        ],
        name="data_augmentation",
    )


def normalize_image(image: tf.Tensor, label: tf.Tensor):
    """Normalize image pixels to [0, 1] and labels to float32."""
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    return image, label


def build_preprocessing_pipeline(
    dataset: tf.data.Dataset,
    augment: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Normalize, optionally cache, augment, and prefetch a dataset.

    Cache is intentionally applied before augmentation, and disabled for
    the augmented training pipeline by default, so random augmentation is
    not frozen into a single cached version.
    """
    num_parallel_calls = config.DATASET_NUM_PARALLEL_CALLS
    dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)

    if cache:
        dataset = dataset.cache()

    if augment:
        augmentation_layer = get_augmentation_layer()

        def _augment(image, label):
            return augmentation_layer(image, training=True), label

        dataset = dataset.map(_augment, num_parallel_calls=num_parallel_calls)
        logger.info("Training augmentation enabled.")

    return dataset.prefetch(config.PREFETCH_BUFFER_SIZE)


def preprocess_single_image(
    image_path: str,
    image_size: tuple[int, int] = config.IMAGE_SIZE,
) -> tf.Tensor:
    """Load one image file and resize it to the model's expected input size."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    raw = tf.io.read_file(image_path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)
