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
        "--phase",
        action="append",
        default=None,
        help=(
            "Build only this configured phase. Repeat for multiple phases; "
            "omit to build every phase."
        ),
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

    defaults = project_config.get("tokenized_dataset_defaults")
    phase_configs = project_config.get("tokenized_datasets")
    if not isinstance(defaults, dict) or not isinstance(phase_configs, dict):
        raise ValueError(
            "configs/data.yaml must contain tokenized_dataset_defaults and "
            "tokenized_datasets mappings."
        )

    if arguments.max_documents is not None:
        if arguments.max_documents < 1:
            raise ValueError("--max-documents must be at least one.")

    selected_phases = arguments.phase or list(phase_configs)
    unknown_phases = set(selected_phases) - set(phase_configs)
    if unknown_phases:
        raise ValueError(
            "Unknown tokenization phases: " + ", ".join(sorted(unknown_phases))
        )

    logger = logging.getLogger(__name__)
    total_sequences = 0
    for phase in selected_phases:
        phase_config = phase_configs[phase]
        if not isinstance(phase_config, dict):
            raise TypeError(f"Tokenized phase '{phase}' must be a mapping.")
        effective_config = {**defaults, **phase_config}
        if arguments.overwrite:
            effective_config["overwrite"] = True
        if arguments.max_documents is not None:
            effective_config["max_documents"] = arguments.max_documents

        logger.info("Building tokenized phase '%s'.", phase)
        manifest = build_tokenized_dataset(effective_config)
        phase_sequences = manifest["totals"]["sequences"]
        total_sequences += phase_sequences
        logger.info("Built %s: sequences=%d", phase, phase_sequences)

    logger.info(
        "Created %d packed sequences across %d phases.",
        total_sequences,
        len(selected_phases),
    )


if __name__ == "__main__":
    main()
