"""Run language verification across configured datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.language.detector import (
    FastTextLanguageDetector,
)
from medical_slm.data.language.pipeline import (
    run_language_verification,
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


def run_configured_language_verification(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run the configured language-verification pipeline."""
    config = load_config(config_path)

    try:
        language_config = config[
            "language_verification"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'language_verification'."
        ) from error

    required_fields = {
        "model_path",
        "output_directory",
        "priority",
    }

    missing_fields = (
        required_fields - language_config.keys()
    )

    if missing_fields:
        missing = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            "Language-verification configuration is "
            f"missing fields: {missing}"
        )

    detector = FastTextLanguageDetector(
        Path(language_config["model_path"]),
        max_sample_characters=int(
            language_config.get(
                "max_sample_characters",
                5000,
            )
        ),
    )

    priority = (
        build_stage_priority(config, input_directory="datasets/interim/near_deduplicated")
        if language_config.get("auto_include_configured_datasets", False)
        else language_config["priority"]
    )
    return run_language_verification(
        priority=priority,
        output_directory=Path(
            language_config["output_directory"]
        ),
        detector=detector,
        config=language_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify and filter document language "
            "using fastText."
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
    """Run language verification."""
    configure_logging()
    arguments = parse_arguments()

    summary = run_configured_language_verification(
        config_path=arguments.config,
    )

    LOGGER.info(
        "Language verification completed: "
        "retained=%d rejected=%d",
        summary["output_documents"],
        summary["rejected_documents"],
    )


if __name__ == "__main__":
    main()
