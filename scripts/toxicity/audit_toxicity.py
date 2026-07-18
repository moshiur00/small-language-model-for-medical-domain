"""Run configured toxicity and safety auditing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.toxicity.detector import (
    TransformersToxicityDetector,
)
from medical_slm.data.toxicity.pipeline import (
    run_toxicity_audit,
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


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load project YAML configuration."""
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


def run_configured_toxicity_audit(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run toxicity auditing from data.yaml."""
    config = load_config(
        config_path
    )

    try:
        toxicity_config = config[
            "toxicity_audit"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'toxicity_audit'."
        ) from error

    required_fields = {
        "model_name",
        "output_directory",
        "priority",
    }

    missing = (
        required_fields
        - toxicity_config.keys()
    )

    if missing:
        raise ValueError(
            "Toxicity-audit configuration is "
            "missing fields: "
            f"{', '.join(sorted(missing))}"
        )

    detector = (
        TransformersToxicityDetector(
            model_name=str(
                toxicity_config[
                    "model_name"
                ]
            ),
            device=str(
                toxicity_config.get(
                    "device",
                    "auto",
                )
            ),
            max_length=int(
                toxicity_config.get(
                    "max_length",
                    512,
                )
            ),
            max_chunks_per_document=int(
                toxicity_config.get(
                    "max_chunks_per_document",
                    5,
                )
            ),
            chunk_stride_tokens=int(
                toxicity_config.get(
                    "chunk_stride_tokens",
                    128,
                )
            ),
            batch_size=int(
                toxicity_config.get(
                    "batch_size",
                    8,
                )
            ),
        )
    )

    return run_toxicity_audit(
        priority=toxicity_config[
            "priority"
        ],
        output_directory=Path(
            toxicity_config[
                "output_directory"
            ]
        ),
        detector=detector,
        config=toxicity_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit toxicity and safety "
            "signals across the corpus."
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
    """Run configured toxicity auditing."""
    configure_logging()

    arguments = parse_arguments()

    summary = (
        run_configured_toxicity_audit(
            config_path=arguments.config,
        )
    )

    LOGGER.info(
        "Toxicity audit completed: "
        "retained=%d review=%d rejected=%d",
        summary["output_documents"],
        summary["review_documents"],
        summary["rejected_documents"],
    )


if __name__ == "__main__":
    main()