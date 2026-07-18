"""Run configured quality scoring and filtering."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.quality.pipeline import (
    run_quality_filtering,
)


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
    """Load YAML project configuration."""
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


def run_configured_quality_filtering(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run quality filtering from data.yaml."""
    config = load_config(config_path)

    try:
        quality_config = config["quality_filtering"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'quality_filtering'."
        ) from error

    required_fields = {
        "output_directory",
        "priority",
    }

    missing_fields = (
        required_fields - quality_config.keys()
    )

    if missing_fields:
        missing = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            "Quality-filtering configuration is "
            f"missing fields: {missing}"
        )

    return run_quality_filtering(
        priority=quality_config["priority"],
        output_directory=Path(
            quality_config["output_directory"]
        ),
        config=quality_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score and filter documents using "
            "interpretable quality metrics."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Run quality filtering."""
    configure_logging()
    arguments = parse_arguments()

    summary = run_configured_quality_filtering(
        config_path=arguments.config,
    )

    LOGGER.info(
        "Quality filtering completed: "
        "retained=%d review=%d rejected=%d",
        summary["output_documents"],
        summary["review_documents"],
        summary["rejected_documents"],
    )


if __name__ == "__main__":
    main()