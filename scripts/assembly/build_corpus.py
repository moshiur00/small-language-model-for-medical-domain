"""Build the final validated model-training corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.assembly.corpus import (
    build_final_corpus,
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


def run_configured_corpus_assembly(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Run corpus assembly using the configured inputs."""
    config = load_config(
        config_path
    )

    try:
        assembly_config = config[
            "corpus_assembly"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'corpus_assembly'."
        ) from error

    required_fields = {
        "output_directory",
        "inputs",
    }

    missing_fields = (
        required_fields
        - assembly_config.keys()
    )

    if missing_fields:
        raise ValueError(
            "Corpus assembly configuration is "
            "missing fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    return build_final_corpus(
        output_directory=Path(
            assembly_config[
                "output_directory"
            ]
        ),
        inputs=assembly_config[
            "inputs"
        ],
        config=assembly_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Assemble final train, validation, "
            "test and tokenizer corpora."
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
    """Build the final corpus."""
    configure_logging()

    arguments = parse_arguments()

    manifest = (
        run_configured_corpus_assembly(
            config_path=arguments.config,
        )
    )

    split_reports = manifest[
        "split_reports"
    ]

    LOGGER.info(
        "Corpus assembly completed: "
        "train=%d validation=%d test=%d",
        split_reports[
            "train"
        ]["output_documents"],
        split_reports[
            "validation"
        ]["output_documents"],
        split_reports[
            "test"
        ]["output_documents"],
    )


if __name__ == "__main__":
    main()