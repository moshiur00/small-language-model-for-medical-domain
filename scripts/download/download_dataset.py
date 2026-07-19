"""Download one configured dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from medical_slm.data.download import Standardizer, download_dataset
from medical_slm.data.standardizers import (
    standardize_alpaca,
    standardize_chatdoctor,
    standardize_fineweb_edu,
    standardize_medalpaca,
    standardize_medinstruct,
    standardize_medmcqa,
    standardize_openmedinstruct,
    standardize_pmc_open_access,
    standardize_project_gutenberg_public_domain,
    standardize_pubmed_abstracts,
    standardize_pubmedqa,
    standardize_tinystories,
    standardize_wikidoc,
    standardize_wikipedia,
    standardize_wikitext,
)


LOGGER = logging.getLogger(__name__)


STANDARDIZERS: dict[str, Standardizer] = {
    "tinystories": standardize_tinystories,
    "wikitext103": standardize_wikitext,
    "wikipedia": standardize_wikipedia,
    "fineweb_edu": standardize_fineweb_edu,
    "project_gutenberg_public_domain": standardize_project_gutenberg_public_domain,
    "pubmed_abstracts": standardize_pubmed_abstracts,
    "pmc_open_access": standardize_pmc_open_access,
    "wikidoc": standardize_wikidoc,
    "medmcqa": standardize_medmcqa,
    "pubmedqa": standardize_pubmedqa,
    "alpaca": standardize_alpaca,
    "medalpaca": standardize_medalpaca,
    "medinstruct": standardize_medinstruct,
    "chatdoctor": standardize_chatdoctor,
    "openmedinstruct": standardize_openmedinstruct,
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
        choices=["all", *sorted(STANDARDIZERS)],
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

    dataset_names = (
        list(STANDARDIZERS) if arguments.dataset == "all" else [arguments.dataset]
    )
    for dataset_name in dataset_names:
        split_counts = download_dataset(
            config_path=arguments.config,
            dataset_name=dataset_name,
            standardizer=STANDARDIZERS[dataset_name],
        )
        LOGGER.info(
            "Completed dataset=%s split_counts=%s",
            dataset_name,
            split_counts,
        )


if __name__ == "__main__":
    main()
