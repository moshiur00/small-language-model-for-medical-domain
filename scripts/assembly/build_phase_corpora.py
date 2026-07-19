"""Build tokenizer, Stage A, continual-pretraining, and SFT corpora."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from medical_slm.data.assembly.phases import build_phase_corpora
from medical_slm.data.tokenization.pipeline import encode_without_special_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpora.yaml"))
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("datasets/interim/toxicity_audited"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("datasets/processed/corpora"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "Use this trained tokenizer for exact quota accounting. "
            "Each document includes one EOS separator token."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Only rebuild the named corpus phases.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token_counter = None
    if arguments.tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            arguments.tokenizer_path,
            use_fast=True,
            local_files_only=True,
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token.")

        def count_stream_tokens(text: str) -> int:
            return len(encode_without_special_tokens(tokenizer, text)) + 1

        token_counter = count_stream_tokens

    manifest = build_phase_corpora(
        corpora_config_path=arguments.config,
        input_directory=arguments.input_directory,
        output_directory=arguments.output_directory,
        token_counter=token_counter,
        tokenizer_path=arguments.tokenizer_path,
        selected_phases=arguments.phases,
    )
    for phase, report in manifest["phases"].items():
        logging.info("Built %s: documents=%d", phase, report["documents"])


if __name__ == "__main__":
    main()
