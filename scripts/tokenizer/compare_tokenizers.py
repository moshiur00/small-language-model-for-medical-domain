"""Compare the custom tokenizer against GPT-2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.tokenizer.compare import (
    DEFAULT_GPT2_TOKENIZER_NAME,
    compare_tokenizers,
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a custom ByteLevel BPE "
            "tokenizer against GPT-2."
        )
    )


    parser.add_argument(
    "--config",
    type=Path,
    default=Path(
        "configs/data.yaml"
    ),
    help="Project YAML configuration path.",
)

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help=(
            "Override the configured maximum "
            "number of evaluation documents."
        ),
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "Load GPT-2 only from the local "
            "Hugging Face cache."
        ),
    )

    return parser.parse_args()


def load_yaml_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
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


def get_comparison_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the tokenizer-comparison section."""
    tokenizer_config = config.get(
        "tokenizer",
        config,
    )

    if not isinstance(
        tokenizer_config,
        dict,
    ):
        raise TypeError(
            "tokenizer configuration must "
            "be a mapping."
        )

    comparison_config = tokenizer_config.get(
        "comparison",
        {},
    )

    if not isinstance(
        comparison_config,
        dict,
    ):
        raise TypeError(
            "tokenizer.comparison must "
            "be a mapping."
        )

    return {
        **tokenizer_config,
        "comparison": comparison_config,
    }


def resolve_evaluation_files(
    comparison_config: dict[str, Any],
) -> list[Path]:
    """Resolve configured evaluation JSONL files."""
    raw_files = comparison_config.get(
        "evaluation_files"
    )

    if not isinstance(raw_files, list):
        raise ValueError(
            "tokenizer.comparison.evaluation_files "
            "must be a list."
        )

    files = [
        Path(str(path))
        for path in raw_files
    ]

    if not files:
        raise ValueError(
            "At least one evaluation file "
            "must be configured."
        )

    return files


def main() -> None:
    """Run tokenizer comparison."""
    configure_logging()
    arguments = parse_arguments()

    config = load_yaml_config(
        arguments.config
    )

    tokenizer_config = (
        get_comparison_config(config)
    )

    comparison_config = (
        tokenizer_config["comparison"]
    )

    configured_max_documents = (
        comparison_config.get(
            "max_documents"
        )
    )

    max_documents = (
        arguments.max_documents
        if arguments.max_documents
        is not None
        else configured_max_documents
    )

    medical_terms = (
        comparison_config.get(
            "medical_terms"
        )
    )

    if (
        medical_terms is not None
        and not isinstance(
            medical_terms,
            list,
        )
    ):
        raise TypeError(
            "medical_terms must be a list."
        )

    result = compare_tokenizers(
        custom_tokenizer_path=Path(
            str(
                comparison_config.get(
                    "custom_tokenizer_path",
                    tokenizer_config.get(
                        "output_directory",
                        "artifacts/tokenizer",
                    ),
                )
            )
        ),
        evaluation_files=(
            resolve_evaluation_files(
                comparison_config
            )
        ),
        output_directory=Path(
            str(
                comparison_config.get(
                    "output_directory",
                    (
                        "artifacts/tokenizer/"
                        "comparison"
                    ),
                )
            )
        ),
        gpt2_tokenizer_name=str(
            comparison_config.get(
                "gpt2_tokenizer_name",
                DEFAULT_GPT2_TOKENIZER_NAME,
            )
        ),
        text_field=str(
            comparison_config.get(
                "text_field",
                "text",
            )
        ),
        max_documents=(
            int(max_documents)
            if max_documents is not None
            else None
        ),
        medical_terms=(
            [
                str(term)
                for term in medical_terms
            ]
            if medical_terms is not None
            else None
        ),
        sample_count=int(
            comparison_config.get(
                "sample_count",
                5,
            )
        ),
        local_files_only=(
            arguments.local_files_only
        ),
    )

    LOGGER.info(
        "Recommended tokenizer: %s",
        result.recommendation[
            "selected_tokenizer"
        ],
    )

    LOGGER.info(
        "JSON report: %s",
        (
            Path(
                str(
                    comparison_config.get(
                        "output_directory",
                        (
                            "artifacts/tokenizer/"
                            "comparison"
                        ),
                    )
                )
            )
            / "tokenizer_comparison.json"
        ),
    )

    LOGGER.info(
        "Markdown report: %s",
        (
            Path(
                str(
                    comparison_config.get(
                        "output_directory",
                        (
                            "artifacts/tokenizer/"
                            "comparison"
                        ),
                    )
                )
            )
            / "tokenizer_comparison.md"
        ),
    )


if __name__ == "__main__":
    main()