"""Build packed binary pretraining datasets from processed JSONL files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.tokenization.pipeline import build_tokenized_dataset


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize and pack train/validation/test corpora."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Project YAML configuration file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing tokenized dataset before rebuilding it.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional development limit applied to every split.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping.")

    return config


def main() -> None:
    """Run tokenized-dataset construction."""
    configure_logging()
    arguments = parse_arguments()
    project_config = load_config(arguments.config)

    tokenized_config = project_config.get("tokenized_dataset")
    if not isinstance(tokenized_config, dict):
        raise ValueError(
            "configs/data.yaml must contain a tokenized_dataset mapping."
        )

    effective_config = dict(tokenized_config)

    if arguments.overwrite:
        effective_config["overwrite"] = True

    if arguments.max_documents is not None:
        if arguments.max_documents < 1:
            raise ValueError("--max-documents must be at least one.")
        effective_config["max_documents"] = arguments.max_documents

    manifest = build_tokenized_dataset(effective_config)

    logging.getLogger(__name__).info(
        "Created %d packed sequences.",
        manifest["totals"]["sequences"],
    )


if __name__ == "__main__":
    main()
