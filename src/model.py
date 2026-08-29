"""
EfficientNet transfer-learning model for CIFAKE binary classification.

The public functions in this module intentionally keep their previous names
(`build_cnn_model`, `compile_model`, `load_trained_model`) so the existing
Flask app and command-line scripts can keep importing them unchanged.
"""

import os
from typing import Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from src import config
from src.utils import get_logger, validate_model_file

logger = get_logger(__name__)


def build_efficientnet_model(
    input_shape=config.INPUT_SHAPE,
    dropout_rate: float = config.HEAD_DROPOUT_RATE,
    dense_units: int = config.HEAD_DENSE_UNITS,
    weights: Optional[str] = config.IMAGENET_WEIGHTS,
) -> tf.keras.Model:
    """
    Build an EfficientNetB0 sigmoid classifier.

    The existing preprocessing path normalizes images to [0, 1]. Keras
    EfficientNet pretrained weights expect the original [0, 255] value range,
    so this model scales tensors back internally. That preserves the current
    prediction API and keeps old Flask code compatible.
    """
    if len(input_shape) != 3:
        raise ValueError("input_shape must be (height, width, channels).")
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("dropout_rate must be in the range [0, 1).")
    if dense_units <= 0:
        raise ValueError("dense_units must be positive.")

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Lambda(lambda image: image * 255.0, name="efficientnet_value_range")(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.Conv2D(
        filters=128,
        kernel_size=(1, 1),
        padding="same",
        activation="swish",
        name=config.GRADCAM_LAST_CONV_LAYER,
    )(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="classifier_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    x = layers.Dense(dense_units, activation="swish", name="classifier_dense")(x)
    x = layers.Dropout(dropout_rate / 2, name="classifier_dense_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CIFAKE_EfficientNetB0")
    logger.info(
        "Built %s transfer-learning model with input_shape=%s and weights=%s.",
        config.MODEL_BACKBONE,
        input_shape,
        weights,
    )
    return model


def build_cnn_model(*args, **kwargs) -> tf.keras.Model:
    """
    Backward-compatible alias for the new transfer-learning model.

    Older scripts in this project imported `build_cnn_model`; retaining the
    name avoids breaking them while replacing the architecture.
    """
    if args and "input_shape" not in kwargs and isinstance(args[0], tuple):
        kwargs["input_shape"] = args[0]
        args = args[1:]

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in {"input_shape", "dropout_rate", "dense_units", "weights"}
    }
    if isinstance(supported_kwargs.get("dense_units"), (list, tuple)):
        dense_units = supported_kwargs["dense_units"]
        supported_kwargs["dense_units"] = dense_units[0] if dense_units else config.HEAD_DENSE_UNITS
    if args:
        logger.warning("Ignoring legacy positional CNN arguments: %s", args)
    ignored_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if ignored_kwargs:
        logger.warning("Ignoring legacy CNN keyword arguments: %s", ignored_kwargs)
    return build_efficientnet_model(**supported_kwargs)


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """Compile a Keras binary classifier with Adam and BCE loss."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=config.LOSS_FUNCTION,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    logger.info("Model compiled with Adam learning_rate=%s.", learning_rate)
    return model


def set_backbone_trainable(
    model: tf.keras.Model,
    trainable: bool,
    fine_tune_last_layers: int = config.FINE_TUNE_LAST_LAYERS,
) -> tf.keras.Model:
    """
    Freeze or partially unfreeze the EfficientNet backbone.

    BatchNormalization layers remain frozen during fine-tuning. This is a
    standard transfer-learning safeguard that reduces overfitting and keeps
    ImageNet statistics stable for small or shifted datasets.
    """
    gradcam_layer_index = next(
        index
        for index, layer in enumerate(model.layers)
        if layer.name == config.GRADCAM_LAST_CONV_LAYER
    )
    backbone_layers = [
        layer
        for layer in model.layers[:gradcam_layer_index]
        if layer.name not in {"input_image", "efficientnet_value_range"}
    ]

    if not trainable:
        for layer in backbone_layers:
            layer.trainable = False
        logger.info("EfficientNet backbone frozen.")
        return model

    for layer in backbone_layers:
        layer.trainable = False

    trainable_layers = [
        layer
        for layer in backbone_layers[-fine_tune_last_layers:]
        if not isinstance(layer, layers.BatchNormalization)
    ]
    for layer in trainable_layers:
        layer.trainable = True

    logger.info(
        "Fine-tuning enabled for %d non-BatchNorm layers from the end of EfficientNet.",
        len(trainable_layers),
    )
    return model


def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load a saved `.keras` model."""
    validate_model_file(model_path)
    try:
        # Loading the full `.keras` artifact can crash native Windows
        # TensorFlow/Python 3.13 while deserializing the saved Lambda layer.
        # Rebuild the known architecture and load only the saved weights; this
        # preserves inference behaviour without executing serialized Lambda
        # config during startup.
        model = build_efficientnet_model(weights=None)
        model.load_weights(model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not load model from '{model_path}': {exc}") from exc

    logger.info("Loaded model from %s.", os.path.abspath(model_path))
    return model


if __name__ == "__main__":
    transfer_model = compile_model(build_efficientnet_model())
    transfer_model.summary()
