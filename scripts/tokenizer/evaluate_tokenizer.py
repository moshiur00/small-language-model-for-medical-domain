"""Evaluate the configured tokenizer."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from medical_slm.tokenizer.evaluate import (
    evaluate_tokenizer,
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


def run_configured_tokenizer_evaluation(
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Evaluate the tokenizer using configured corpora."""
    config = load_config(config_path)

    try:
        tokenizer_config = config[
            "tokenizer"
        ]
        evaluation_config = (
            tokenizer_config["evaluation"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Configuration must contain "
            "'tokenizer.evaluation'."
        ) from error

    inputs = evaluation_config.get(
        "inputs"
    )

    if (
        not isinstance(inputs, Sequence)
        or isinstance(inputs, str)
    ):
        raise TypeError(
            "tokenizer.evaluation.inputs "
            "must be a sequence."
        )

    for index, entry in enumerate(inputs):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Evaluation input {index} "
                "must be a mapping."
            )

        missing = {
            "name",
            "path",
        } - entry.keys()

        if missing:
            raise ValueError(
                f"Evaluation input {index} "
                "is missing fields: "
                f"{', '.join(sorted(missing))}"
            )

    return evaluate_tokenizer(
        tokenizer_directory=Path(
            tokenizer_config[
                "output_directory"
            ]
        ),
        evaluation_inputs=inputs,
        output_path=Path(
            evaluation_config[
                "output_path"
            ]
        ),
        medical_terms=[
            str(term)
            for term in (
                evaluation_config.get(
                    "medical_terms",
                    [],
                )
            )
        ],
        unicode_normalization=str(
            tokenizer_config.get(
                "unicode_normalization",
                "NFKC",
            )
        ),
        store_sample_encodings=bool(
            evaluation_config.get(
                "store_sample_encodings",
                True,
            )
        ),
        maximum_sample_encodings=int(
            evaluation_config.get(
                "maximum_sample_encodings",
                20,
            )
        ),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tokenizer compression, "
            "coverage and round-trip behavior."
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
    """Evaluate the configured tokenizer."""
    configure_logging()

    arguments = parse_arguments()

    metrics = (
        run_configured_tokenizer_evaluation(
            config_path=arguments.config,
        )
    )

    overall = metrics["overall"]

    LOGGER.info(
        "Tokenizer evaluation completed: "
        "documents=%d tokens_per_word=%.4f "
        "characters_per_token=%.4f unk_rate=%.8f",
        overall["document_count"],
        overall["tokens_per_word"],
        overall["characters_per_token"],
        overall["unknown_token_rate"],
    )


if __name__ == "__main__":
    main()