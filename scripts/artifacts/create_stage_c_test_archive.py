"""Create the physically separate, checksum-addressed Stage C sealed-test archive."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.artifacts.create_stage_c_data_archive import create_archive
except ModuleNotFoundError:  # Direct ``python scripts/artifacts/...`` execution.
    from create_stage_c_data_archive import create_archive


SEALED_TEST_INPUTS = (
    Path("artifacts/tokenizer"),
    Path("datasets/tokenized/sft_stage_c_v1/manifest.json"),
    Path("datasets/tokenized/sft_stage_c_v1/test"),
    Path("datasets/tokenized/evaluation/dataset_manifest.json"),
    Path("datasets/tokenized/evaluation/test"),
    Path("datasets/tokenized/evaluation_medical/dataset_manifest.json"),
    Path("datasets/tokenized/evaluation_medical/test"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/exports/stage-c-sealed-test.tar"),
    )
    return parser.parse_args()


def create_sealed_archive(output: Path) -> tuple[Path, Path]:
    """Create an archive containing test artifacts and no training split."""
    return create_archive(output, inputs=SEALED_TEST_INPUTS)


def main() -> None:
    archive, checksum = create_sealed_archive(parse_arguments().output)
    print({
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": checksum.read_text(encoding="utf-8").split()[0],
        "checksum": str(checksum),
        "contains_training_split": False,
    })


if __name__ == "__main__":
    main()
