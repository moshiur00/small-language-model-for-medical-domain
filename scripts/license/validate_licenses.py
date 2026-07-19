"""Run configured dataset-license validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.licensing.pipeline import (
    run_license_validation,
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


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load YAML project configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            "Configuration file does not exist: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config


def run_configured_license_validation(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run license validation from data.yaml."""
    config = load_config(
        config_path
    )

    try:
        license_config = config[
            "license_validation"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'license_validation'."
        ) from error

    required_fields = {
        "output_directory",
        "priority",
    }

    missing_fields = (
        required_fields
        - license_config.keys()
    )

    if missing_fields:
        raise ValueError(
            "License-validation configuration "
            "is missing fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    priority = (
        build_stage_priority(config, input_directory="datasets/interim/quality_filtered")
        if license_config.get("auto_include_configured_datasets", False)
        else license_config["priority"]
    )
    return run_license_validation(
        priority=priority,
        output_directory=Path(
            license_config[
                "output_directory"
            ]
        ),
        config=license_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate dataset licenses "
            "using configured metadata policies."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/data.yaml"
        ),
        help="Path to data configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Run configured license validation."""
    configure_logging()

    arguments = parse_arguments()

    summary = (
        run_configured_license_validation(
            config_path=arguments.config,
        )
    )

    LOGGER.info(
        "License validation completed: "
        "retained=%d review=%d rejected=%d",
        summary["output_documents"],
        summary["review_documents"],
        summary["rejected_documents"],
    )


if __name__ == "__main__":
    main()
