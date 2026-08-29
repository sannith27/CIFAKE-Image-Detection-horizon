"""
Two-phase training pipeline for the CIFAKE EfficientNetB0 model.
"""

import datetime
import os
from typing import Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from src import config
from src.data_loader import CIFAKEDataLoader
from src.evaluate import evaluate_model, plot_training_history
from src.model import (
    build_efficientnet_model,
    compile_model,
    load_trained_model,
    set_backbone_trainable,
)
from src.utils import check_gpu, get_logger, set_random_seed, timeit

logger = get_logger(__name__)


def _split_epochs(total_epochs: int) -> tuple[int, int]:
    """Split requested epochs across frozen-head training and fine-tuning."""
    if total_epochs <= 1:
        return total_epochs, 0

    configured_total = config.PHASE1_EPOCHS + config.PHASE2_EPOCHS
    phase1_ratio = config.PHASE1_EPOCHS / configured_total
    phase1_epochs = max(1, min(total_epochs - 1, round(total_epochs * phase1_ratio)))
    phase2_epochs = total_epochs - phase1_epochs
    return phase1_epochs, phase2_epochs


def build_callbacks(
    phase_name: str,
    tensorboard_subdir: Optional[str] = None,
    append_history: bool = False,
) -> list:
    """Create Keras callbacks for robust training."""
    run_id = tensorboard_subdir or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, run_id, phase_name)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.CHECKPOINT_MONITOR,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.CHECKPOINT_PATH,
            monitor=config.CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir),
        tf.keras.callbacks.CSVLogger(config.HISTORY_CSV_PATH, append=append_history),
    ]


def _fit_phase(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    phase_name: str,
    initial_epoch: int,
    epochs: int,
    run_id: str,
    append_history: bool,
) -> tf.keras.callbacks.History:
    """Train one phase and return its Keras history."""
    if epochs <= initial_epoch:
        logger.info("Skipping %s because no epochs are allocated.", phase_name)
        return tf.keras.callbacks.History()

    logger.info(
        "Starting %s: initial_epoch=%d, epochs=%d.",
        phase_name,
        initial_epoch,
        epochs,
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=build_callbacks(
            phase_name=phase_name,
            tensorboard_subdir=run_id,
            append_history=append_history,
        ),
        verbose=1,
    )


@timeit
def run_training(
    epochs: int = config.EPOCHS,
    resume_from: Optional[str] = None,
) -> Tuple[tf.keras.Model, Optional[tf.keras.callbacks.History]]:
    """
    Load data, train the model, evaluate it, and save the final `.keras` file.

    Phase 1 trains the classifier head while EfficientNet is frozen. Phase 2
    unfreezes the last EfficientNet layers and fine-tunes with a very small
    learning rate.
    """
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    set_random_seed(config.RANDOM_SEED)
    check_gpu()

    loader = CIFAKEDataLoader()
    train_ds, val_ds, test_ds = loader.load_datasets(augment_train=True)
    phase1_epochs, phase2_epochs = _split_epochs(epochs)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if resume_from:
        logger.info("Resuming training from %s.", resume_from)
        model = load_trained_model(resume_from)
    else:
        model = build_efficientnet_model()

    set_backbone_trainable(model, trainable=False)
    compile_model(model, learning_rate=config.LEARNING_RATE)
    model.summary(print_fn=logger.info)

    phase1_history = _fit_phase(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        phase_name="phase1_frozen_backbone",
        initial_epoch=0,
        epochs=phase1_epochs,
        run_id=run_id,
        append_history=False,
    )

    final_history = phase1_history
    if phase2_epochs > 0:
        set_backbone_trainable(
            model,
            trainable=True,
            fine_tune_last_layers=config.FINE_TUNE_LAST_LAYERS,
        )
        compile_model(model, learning_rate=config.FINE_TUNE_LEARNING_RATE)
        final_history = _fit_phase(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            phase_name="phase2_fine_tuning",
            initial_epoch=phase1_epochs,
            epochs=phase1_epochs + phase2_epochs,
            run_id=run_id,
            append_history=True,
        )

    os.makedirs(os.path.dirname(config.FINAL_MODEL_PATH), exist_ok=True)
    model.save(config.FINAL_MODEL_PATH)
    logger.info("Final model saved to %s.", config.FINAL_MODEL_PATH)

    logger.info("Running automatic evaluation on the test split.")
    metrics = evaluate_model(model, test_ds, save_plots=True)
    for metric_name, value in metrics.items():
        logger.info("Final test %s: %.4f", metric_name, value)
    plot_training_history()

    return model, None if phase2_epochs > 0 else final_history


if __name__ == "__main__":
    run_training()
