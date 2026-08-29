"""
TensorFlow dataset loading for the official CIFAKE dataset.

Official dataset source:
    https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Expected official layout after extraction:
    train/
        FAKE/
        REAL/
    test/
        FAKE/
        REAL/

The loader also tolerates one extra wrapper directory, for example:
    cifake-real-and-ai-generated-synthetic-images/
        train/
            FAKE/
            REAL/
        test/
            FAKE/
            REAL/
"""

import os
from typing import Dict, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

from src import config
from src.preprocessing import build_preprocessing_pipeline
from src.utils import count_images_per_class, get_logger

logger = get_logger(__name__)


class CIFAKEDataLoader:
    """Load CIFAKE train, validation, and test datasets."""

    def __init__(
        self,
        dataset_dir: str = config.DATASET_DIR,
        train_dir: str = config.TRAIN_DIR,
        test_dir: str = config.TEST_DIR,
        image_size: Tuple[int, int] = config.IMAGE_SIZE,
        batch_size: int = config.BATCH_SIZE,
        validation_split: float = config.VALIDATION_SPLIT,
        seed: int = config.RANDOM_SEED,
    ):
        self.dataset_dir = dataset_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.class_names = config.CLASS_NAMES

    @staticmethod
    def _find_case_insensitive_child(parent: str, child_name: str) -> str | None:
        """Return a child directory path matching child_name case-insensitively."""
        if not os.path.isdir(parent):
            return None

        exact_path = os.path.join(parent, child_name)
        if os.path.isdir(exact_path):
            return exact_path

        child_name_lower = child_name.lower()
        for item in os.listdir(parent):
            item_path = os.path.join(parent, item)
            if os.path.isdir(item_path) and item.lower() == child_name_lower:
                return item_path
        return None

    def _looks_like_cifake_root(self, root: str) -> bool:
        """Check whether root contains train/test with FAKE/REAL subfolders."""
        train_dir = self._find_case_insensitive_child(root, "train")
        test_dir = self._find_case_insensitive_child(root, "test")
        if not train_dir or not test_dir:
            return False

        for split_dir in (train_dir, test_dir):
            if not self._find_case_insensitive_child(split_dir, "FAKE"):
                return False
            if not self._find_case_insensitive_child(split_dir, "REAL"):
                return False
        return True

    def _resolve_dataset_root(self) -> str:
        """
        Resolve the CIFAKE root.

        The preferred root is config.DATASET_DIR. If that folder contains one
        extracted wrapper directory, this method finds the nested official
        train/test folders automatically.
        """
        if self._looks_like_cifake_root(self.dataset_dir):
            return self.dataset_dir

        if os.path.isdir(self.dataset_dir):
            for item in os.listdir(self.dataset_dir):
                candidate = os.path.join(self.dataset_dir, item)
                if os.path.isdir(candidate) and self._looks_like_cifake_root(candidate):
                    logger.warning(
                        "Detected CIFAKE inside nested folder: %s. Using it automatically.",
                        candidate,
                    )
                    return candidate

        raise FileNotFoundError(
            "CIFAKE dataset not found. Expected either:\n"
            f"  {os.path.join(self.dataset_dir, 'train', 'FAKE')}\n"
            f"  {os.path.join(self.dataset_dir, 'train', 'REAL')}\n"
            f"  {os.path.join(self.dataset_dir, 'test', 'FAKE')}\n"
            f"  {os.path.join(self.dataset_dir, 'test', 'REAL')}\n"
            "or the same structure inside one nested extracted Kaggle folder."
        )

    def _resolve_split_and_class_names(self, root: str, split_name: str) -> Tuple[str, list[str]]:
        """Resolve split path and actual class folder names while preserving FAKE=0, REAL=1."""
        split_dir = self._find_case_insensitive_child(root, split_name)
        if split_dir is None:
            raise FileNotFoundError(f"Missing CIFAKE split directory: {os.path.join(root, split_name)}")

        class_dirs: Dict[str, str] = {}
        for class_name in config.CLASS_NAMES:
            class_dir = self._find_case_insensitive_child(split_dir, class_name)
            if class_dir is None:
                raise FileNotFoundError(
                    f"Missing class directory {class_name} inside {split_dir}."
                )
            class_dirs[class_name] = os.path.basename(class_dir)

        return split_dir, [class_dirs["FAKE"], class_dirs["REAL"]]

    def _validate_paths(self) -> tuple[str, str, list[str]]:
        """Validate and resolve official CIFAKE dataset paths."""
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1.")

        root = self._resolve_dataset_root()
        self.train_dir, train_class_names = self._resolve_split_and_class_names(root, "train")
        self.test_dir, test_class_names = self._resolve_split_and_class_names(root, "test")

        if [name.lower() for name in train_class_names] != [name.lower() for name in test_class_names]:
            raise ValueError(
                "Train/test class directories do not match. Expected FAKE and REAL in both splits."
            )

        for split_name, split_dir in (("training", self.train_dir), ("test", self.test_dir)):
            counts = count_images_per_class(split_dir)
            logger.info("%s image counts: %s", split_name.capitalize(), counts)
            empty_classes = [name for name, count in counts.items() if count == 0]
            if empty_classes:
                raise ValueError(
                    f"{split_name.capitalize()} dataset has empty class folders: {empty_classes}."
                )

        logger.info("Using CIFAKE root: %s", root)
        logger.info("Using train directory: %s", self.train_dir)
        logger.info("Using test directory: %s", self.test_dir)
        return self.train_dir, self.test_dir, train_class_names

    def load_datasets(
        self, augment_train: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Return train, validation, and test tf.data pipelines."""
        train_dir, test_dir, actual_class_names = self._validate_paths()

        raw_train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            class_names=actual_class_names,
            color_mode="rgb",
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="training",
        )
        raw_val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="binary",
            class_names=actual_class_names,
            color_mode="rgb",
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False,
            seed=self.seed,
            validation_split=self.validation_split,
            subset="validation",
        )
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            class_names=actual_class_names,
            color_mode="rgb",
            image_size=self.image_size,
            batch_size=self.batch_size,
            # Shuffle only when a batch limit is active so a small evaluation
            # sample is unlikely to contain one class exclusively.
            shuffle=config.MAX_TEST_BATCHES > 0,
            seed=self.seed,
        )

        self.class_names = config.CLASS_NAMES
        logger.info("Class mapping: FAKE=0, REAL=1. Actual folders: %s", actual_class_names)

        train_ds = build_preprocessing_pipeline(raw_train_ds, augment=augment_train, cache=False)
        val_ds = build_preprocessing_pipeline(raw_val_ds, augment=False, cache=False)
        test_ds = build_preprocessing_pipeline(raw_test_ds, augment=False, cache=False)

        dataset_limits = (
            ("training", train_ds, config.MAX_TRAIN_BATCHES),
            ("validation", val_ds, config.MAX_VAL_BATCHES),
            ("test", test_ds, config.MAX_TEST_BATCHES),
        )
        limited_datasets = []
        for split_name, dataset, max_batches in dataset_limits:
            if max_batches > 0:
                dataset = dataset.take(max_batches)
                logger.warning(
                    "Limiting %s dataset to %d batches via configuration.",
                    split_name,
                    max_batches,
                )
            limited_datasets.append(dataset)
        train_ds, val_ds, test_ds = limited_datasets

        logger.info("Datasets loaded successfully.")
        return train_ds, val_ds, test_ds


if __name__ == "__main__":
    loader = CIFAKEDataLoader()
    train_ds, _, _ = loader.load_datasets()
    for images, labels in train_ds.take(1):
        logger.info("Batch shape: %s, Labels shape: %s", images.shape, labels.shape)
