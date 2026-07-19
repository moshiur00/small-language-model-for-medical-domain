"""Run global exact deduplication across configured datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.deduplication.global_exact import (
    run_global_exact_deduplication,
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
    """Load the YAML project configuration."""
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


def run_configured_global_deduplication(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run globally configured exact deduplication."""
    config = load_config(config_path)

    try:
        exact_config = config["deduplication"]["exact"]
        global_config = config["deduplication"]["global"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'deduplication.exact' and 'deduplication.global'."
        ) from error

    required_fields = {
        "output_directory",
        "priority",
    }

    missing_fields = (
        required_fields - global_config.keys()
    )

    if missing_fields:
        missing = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            "Global deduplication configuration is missing "
            f"fields: {missing}"
        )

    priority = (
        build_stage_priority(config, input_directory="datasets/interim/deduplicated")
        if global_config.get("auto_include_configured_datasets", False)
        else global_config["priority"]
    )
    return run_global_exact_deduplication(
        priority=priority,
        output_directory=Path(
            global_config["output_directory"]
        ),
        deduplication_config=exact_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform exact deduplication across all "
            "configured datasets and splits."
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
    """Run global exact deduplication."""
    configure_logging()
    arguments = parse_arguments()

    summary = run_configured_global_deduplication(
        config_path=arguments.config,
    )

    LOGGER.info(
        "Completed global deduplication: output=%d",
        summary["output_documents"],
    )


if __name__ == "__main__":
    main()
