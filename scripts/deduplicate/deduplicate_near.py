"""Run global MinHash-based near-duplicate removal."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.deduplication.minhash import (
    run_global_near_deduplication,
)
from medical_slm.data.pipeline_inventory import build_stage_priority


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
    """Load the project configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config


def run_configured_near_deduplication(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run the configured global near-deduplication stage."""
    config = load_config(config_path)

    try:
        near_config = config["deduplication"]["near"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain 'deduplication.near'."
        ) from error

    required_fields = {
        "output_directory",
        "priority",
    }

    missing_fields = required_fields - near_config.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))

        raise ValueError(
            "Near-deduplication configuration is missing "
            f"fields: {missing}"
        )

    priority = (
        build_stage_priority(config, input_directory="datasets/interim/global_deduplicated")
        if near_config.get("auto_include_configured_datasets", False)
        else near_config["priority"]
    )
    return run_global_near_deduplication(
        priority=priority,
        output_directory=Path(
            near_config["output_directory"]
        ),
        config=near_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform global near-duplicate removal "
            "using MinHash and exact Jaccard verification."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run near-duplicate removal."""
    configure_logging()
    arguments = parse_arguments()

    summary = run_configured_near_deduplication(
        config_path=arguments.config,
    )

    LOGGER.info(
        "Completed near-deduplication: retained=%d removed=%d",
        summary["output_documents"],
        summary["near_duplicate_documents"],
    )


if __name__ == "__main__":
    main()
