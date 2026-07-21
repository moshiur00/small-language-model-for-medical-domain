"""Build document-disjoint medical validation and test corpora."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from medical_slm.data.assembly.phases import (
    assemble_disjoint_evaluation_corpora,
    load_document_ids,
)
from medical_slm.data.tokenization.pipeline import encode_without_special_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpora.yaml"))
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("datasets/interim/license_validated"),
    )
    parser.add_argument(
        "--corpora-directory",
        type=Path,
        default=Path("datasets/processed/corpora"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("datasets/processed/evaluation_medical"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/tokenizer"),
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    evaluation_config = config.get("medical_evaluation")
    if not isinstance(evaluation_config, dict):
        raise ValueError("Corpora config does not contain medical_evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")

    def count_stream_tokens(text: str) -> int:
        return len(encode_without_special_tokens(tokenizer, text)) + 1

    excluded_document_ids: set[str] = set()
    excluded_phases = evaluation_config.get("exclude_phases", [])
    for phase in excluded_phases:
        path = arguments.corpora_directory / str(phase) / "train.jsonl"
        excluded_document_ids.update(load_document_ids(path))

    manifest = assemble_disjoint_evaluation_corpora(
        evaluation_config=evaluation_config,
        input_directory=arguments.input_directory,
        output_directory=arguments.output_directory,
        excluded_document_ids=excluded_document_ids,
        characters_per_token=float(config.get("estimated_characters_per_token", 4.0)),
        token_counter=count_stream_tokens,
    )
    logging.info(
        "Built medical evaluation: documents=%d",
        manifest["selected_document_id_count"],
    )


if __name__ == "__main__":
    main()
