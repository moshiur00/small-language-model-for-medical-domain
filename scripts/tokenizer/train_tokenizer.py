"""Train the configured byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.tokenizer.train import (
    train_byte_level_bpe,
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
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    return config


def run_configured_tokenizer_training(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Train the tokenizer from configured corpus files."""
    config = load_config(config_path)

    try:
        tokenizer_config = config[
            "tokenizer"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain 'tokenizer'."
        ) from error

    training_corpus = Path(
        tokenizer_config[
            "training_corpus"
        ]
    )

    output_directory = Path(
        tokenizer_config[
            "output_directory"
        ]
    )

    return train_byte_level_bpe(
        training_files=[
            training_corpus
        ],
        output_directory=output_directory,
        config=tokenizer_config,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a byte-level BPE tokenizer "
            "from the processed training corpus."
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
    """Train the configured tokenizer."""
    configure_logging()

    arguments = parse_arguments()

    result = (
        run_configured_tokenizer_training(
            config_path=arguments.config,
        )
    )

    summary = result["summary"]

    LOGGER.info(
        "Tokenizer trained: "
        "requested_vocab=%d actual_vocab=%d",
        summary[
            "requested_vocabulary_size"
        ],
        summary[
            "trained_vocabulary_size"
        ],
    )


if __name__ == "__main__":
    main()