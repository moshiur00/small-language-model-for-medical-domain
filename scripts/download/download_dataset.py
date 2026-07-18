"""Download one configured dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from medical_slm.data.download import Standardizer, download_dataset
from medical_slm.data.standardizers import (
    standardize_tinystories,
    standardize_wikipedia,
    standardize_wikitext,
)


LOGGER = logging.getLogger(__name__)


STANDARDIZERS: dict[str, Standardizer] = {
    "tinystories": standardize_tinystories,
    "wikitext103": standardize_wikitext,
    "wikipedia": standardize_wikipedia,
}


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and standardize a configured dataset."
    )

    parser.add_argument(
        "dataset",
        choices=sorted(STANDARDIZERS),
        help="Dataset configuration to process.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the data configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the configured dataset ingestion process."""
    configure_logging()
    arguments = parse_arguments()

    standardizer = STANDARDIZERS[arguments.dataset]

    split_counts = download_dataset(
        config_path=arguments.config,
        dataset_name=arguments.dataset,
        standardizer=standardizer,
    )

    LOGGER.info(
        "Completed dataset=%s split_counts=%s",
        arguments.dataset,
        split_counts,
    )


if __name__ == "__main__":
    main()