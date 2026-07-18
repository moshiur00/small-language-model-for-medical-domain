"""Clean one configured raw dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.cleaning.pipeline import clean_jsonl_file


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    if "cleaning" not in config:
        raise ValueError(
            "Configuration must contain a 'cleaning' section."
        )

    return config


def clean_dataset(
    *,
    config_path: Path,
    dataset_name: str,
) -> None:
    """Clean all configured splits for one dataset."""
    config = load_config(config_path)
    cleaning_config = config["cleaning"]

    datasets_config = cleaning_config.get("datasets", {})

    if dataset_name not in datasets_config:
        available = ", ".join(sorted(datasets_config))
        raise KeyError(
            f"Unknown cleaning dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    dataset_config = datasets_config[dataset_name]

    input_directory = Path(dataset_config["input_directory"])
    output_directory = Path(dataset_config["output_directory"])
    splits = dataset_config["splits"]

    output_directory.mkdir(parents=True, exist_ok=True)

    for split in splits:
        input_path = input_directory / f"{split}.jsonl"
        output_path = output_directory / f"{split}.jsonl"

        statistics = clean_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            cleaning_config=cleaning_config,
        )

        LOGGER.info(
            "Completed dataset=%s split=%s input=%d output=%d",
            dataset_name,
            split,
            statistics["input_documents"],
            statistics["output_documents"],
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean a configured raw dataset."
    )

    parser.add_argument(
        "dataset",
        help="Dataset key under cleaning.datasets.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Run dataset cleaning."""
    configure_logging()
    arguments = parse_arguments()

    clean_dataset(
        config_path=arguments.config,
        dataset_name=arguments.dataset,
    )


if __name__ == "__main__":
    main()