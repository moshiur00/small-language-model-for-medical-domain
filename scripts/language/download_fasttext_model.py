"""Download the official fastText language-identification model."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from urllib.request import urlopen


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/"
    "fasttext/supervised-models/lid.176.bin"
)

DEFAULT_OUTPUT_PATH = Path(
    "models/language_identification/lid.176.bin"
)

DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def download_file(
    *,
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Download a remote file to a local path."""
    if output_path.exists() and not overwrite:
        LOGGER.info(
            "Model already exists at %s. Skipping download.",
            output_path,
        )
        return output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    LOGGER.info(
        "Downloading language model from %s",
        url,
    )

    try:
        with urlopen(url) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(
                    response,
                    output_file,
                    length=DOWNLOAD_CHUNK_SIZE,
                )

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    LOGGER.info(
        "Downloaded model to %s",
        output_path,
    )

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the official fastText "
            "lid.176 language-identification model."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="Model download URL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination model path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model file.",
    )

    return parser.parse_args()


def main() -> None:
    """Download the model."""
    configure_logging()
    arguments = parse_arguments()

    download_file(
        url=arguments.url,
        output_path=arguments.output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()