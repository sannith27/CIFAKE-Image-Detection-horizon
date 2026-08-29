"""
Unified command-line entry point for the CIFAKE project.
"""

import argparse
import os
import sys

from src import config
from src.utils import check_gpu, get_logger, set_random_seed

logger = get_logger(__name__)


def _add_device_argument(parser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Execution device: prefer GPU automatically, require GPU, or force CPU.",
    )


def _add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Train the CIFAKE CNN model.")
    _add_device_argument(parser)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--resume-from", type=str, default=None)


def _add_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    _add_device_argument(parser)
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)


def _add_predict_parser(subparsers) -> None:
    parser = subparsers.add_parser("predict", help="Predict REAL/FAKE for image(s).")
    _add_device_argument(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--dir", type=str)
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)


def _add_gradcam_parser(subparsers) -> None:
    parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM for one image.")
    _add_device_argument(parser)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CIFAKE: detect REAL vs AI-generated images and explain predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_predict_parser(subparsers)
    _add_gradcam_parser(subparsers)
    return parser


def main() -> int:
    """Run the requested CLI command."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        os.environ["CIFAKE_DEVICE"] = args.device
        set_random_seed(config.RANDOM_SEED)
        check_gpu()

        if args.command == "train":
            from src.evaluate import plot_training_history
            from src.train import run_training

            _, history = run_training(epochs=args.epochs, resume_from=args.resume_from)
            plot_training_history(history)
            logger.info("Training complete.")

        elif args.command == "evaluate":
            from src.data_loader import CIFAKEDataLoader
            from src.evaluate import evaluate_model, plot_training_history
            from src.model import load_trained_model

            model = load_trained_model(args.model)
            _, _, test_ds = CIFAKEDataLoader().load_datasets(augment_train=False)
            evaluate_model(model, test_ds)
            plot_training_history()
            logger.info("Evaluation complete.")

        elif args.command == "predict":
            from src.model import load_trained_model
            from src.predict import predict_batch, predict_single_image

            model = load_trained_model(args.model)
            result = predict_single_image(model, args.image) if args.image else predict_batch(model, args.dir)
            print(result)

        elif args.command == "gradcam":
            from src.gradcam import explain_image
            from src.model import load_trained_model
            from src.utils import format_confidence

            model = load_trained_model(args.model)
            label, path, confidence = explain_image(model, args.image)
            print(f"Prediction: {label} ({format_confidence(confidence)}) | Grad-CAM overlay saved to: {path}")

        else:
            parser.print_help()
            return 1

        return 0

    except ModuleNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except ValueError as exc:
        logger.error("Invalid input/configuration: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
