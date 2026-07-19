"""Run exact deduplication for one configured dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.deduplication.exact import (
    deduplicate_dataset_splits,
)
from medical_slm.data.pipeline_inventory import configured_splits, split_priority


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the project YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config


def deduplicate_configured_dataset(
    *,
    config_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    """Deduplicate one dataset configured in data.yaml."""
    config = load_config(config_path)

    try:
        exact_config = config["deduplication"]["exact"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'deduplication.exact'."
        ) from error

    datasets_config = exact_config.get(
        "datasets",
        {},
    )

    if dataset_name not in datasets_config and dataset_name in config.get("datasets", {}):
        datasets_config = {
            **datasets_config,
            dataset_name: {
                "input_directory": f"datasets/interim/cleaned/{dataset_name}",
                "output_directory": f"datasets/interim/deduplicated/{dataset_name}",
                "split_priority": list(split_priority(config, dataset_name)),
            },
        }

    if dataset_name not in datasets_config:
        available = ", ".join(
            sorted(datasets_config)
        )

        raise KeyError(
            f"Unknown deduplication dataset "
            f"{dataset_name!r}. Available datasets: {available}"
        )

    dataset_config = datasets_config[dataset_name]

    required_fields = {
        "input_directory",
        "output_directory",
        "split_priority",
    }

    missing_fields = (
        required_fields - dataset_config.keys()
    )

    if missing_fields:
        missing = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            f"Dataset {dataset_name!r} is missing "
            f"configuration fields: {missing}"
        )

    summary = deduplicate_dataset_splits(
        input_directory=Path(
            dataset_config["input_directory"]
        ),
        output_directory=Path(
            dataset_config["output_directory"]
        ),
        split_priority=list(
            dataset_config["split_priority"]
        ),
        deduplication_config=exact_config,
    )

    LOGGER.info(
        "Completed exact deduplication for dataset=%s",
        dataset_name,
    )

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform exact deduplication for a "
            "configured dataset."
        )
    )

    parser.add_argument(
        "dataset",
        help="Dataset key under deduplication.exact.datasets.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the exact-deduplication CLI."""
    configure_logging()
    arguments = parse_arguments()

    if arguments.dataset == "all":
        config = load_config(arguments.config)
        dataset_names = list(dict.fromkeys(dataset for dataset, _ in configured_splits(config)))
    else:
        dataset_names = [arguments.dataset]
    for dataset_name in dataset_names:
        deduplicate_configured_dataset(
            config_path=arguments.config,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    main()
