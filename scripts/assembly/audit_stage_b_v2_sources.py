"""Audit unused general data and Stage A replay needs for Stage B v2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from medical_slm.data.assembly.phases import (
    document_id,
    hash_document_ids,
    load_document_ids,
)
from medical_slm.data.jsonl import read_jsonl
from medical_slm.data.tokenization.pipeline import encode_without_special_tokens


GENERAL_SOURCES = (
    "fineweb_edu",
    "wikipedia",
    "tinystories",
    "wikitext103",
    "project_gutenberg_public_domain",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("datasets/interim/license_validated"),
    )
    parser.add_argument(
        "--stage-a-corpus",
        type=Path,
        default=Path("datasets/processed/corpora/stage_a_325m/train.jsonl"),
    )
    parser.add_argument(
        "--stage-b-v1-corpus",
        type=Path,
        default=Path(
            "datasets/processed/corpora/continual_medical_stage_b_225m/train.jsonl"
        ),
    )
    parser.add_argument(
        "--phase-manifest",
        type=Path,
        default=Path("datasets/processed/corpora/manifest.json"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer"),
    )
    parser.add_argument(
        "--target-medical-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_b/v2/source_audit.json"),
    )
    return parser.parse_args()


def load_phase_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Phase manifest must contain a JSON object.")
    return value


def v1_medical_tokens(manifest: dict[str, Any]) -> int:
    phase = manifest["phases"]["continual_medical_stage_b_225m"]
    return sum(
        int(report["exact_tokens"])
        for report in phase["sources"].values()
        if report["domain"] == "medical"
    )


def main() -> None:
    arguments = parse_arguments()
    medical_fraction = arguments.target_medical_fraction
    if not 0.0 < medical_fraction < 1.0:
        raise ValueError("target-medical-fraction must be between zero and one.")

    stage_a_ids = load_document_ids(arguments.stage_a_corpus)
    stage_b_v1_ids = load_document_ids(arguments.stage_b_v1_corpus)
    overlap = stage_a_ids & stage_b_v1_ids
    if overlap:
        raise ValueError("Stage A and Stage B v1 unexpectedly overlap.")
    used_ids = stage_a_ids | stage_b_v1_ids

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.tokenizer,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS.")

    source_reports: dict[str, dict[str, Any]] = {}
    all_unused_ids: set[str] = set()
    total_unused_tokens = 0
    total_unused_documents = 0

    for source in GENERAL_SOURCES:
        input_path = arguments.input_directory / source / "train.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        available_documents = 0
        used_documents = 0
        unused_documents = 0
        unused_tokens = 0
        unused_ids: set[str] = set()
        for record in read_jsonl(input_path):
            available_documents += 1
            identifier = document_id(record)
            if identifier in used_ids:
                used_documents += 1
                continue
            if identifier in all_unused_ids:
                raise ValueError(f"Duplicate unused document id: {identifier}")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            unused_ids.add(identifier)
            all_unused_ids.add(identifier)
            unused_documents += 1
            unused_tokens += len(encode_without_special_tokens(tokenizer, text)) + 1
        source_reports[source] = {
            "available_documents": available_documents,
            "used_documents": used_documents,
            "unused_documents": unused_documents,
            "unused_exact_tokens": unused_tokens,
            "unused_document_id_sha256": hash_document_ids(unused_ids),
        }
        total_unused_documents += unused_documents
        total_unused_tokens += unused_tokens

    phase_manifest = load_phase_manifest(arguments.phase_manifest)
    medical_tokens = v1_medical_tokens(phase_manifest)
    target_general_tokens = round(
        medical_tokens * (1.0 - medical_fraction) / medical_fraction
    )
    existing_v1_general_tokens = int(
        phase_manifest["phases"]["continual_medical_stage_b_225m"]["sources"][
            "fineweb_edu"
        ]["exact_tokens"]
    )
    additional_general_tokens = max(
        0,
        target_general_tokens - existing_v1_general_tokens,
    )
    replay_tokens_required = max(
        0,
        additional_general_tokens - total_unused_tokens,
    )

    report = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "medical_fraction": medical_fraction,
            "general_fraction": 1.0 - medical_fraction,
            "preserved_v1_medical_tokens": medical_tokens,
            "target_general_tokens": target_general_tokens,
            "existing_v1_general_tokens": existing_v1_general_tokens,
            "additional_general_tokens_required": additional_general_tokens,
        },
        "used_document_ids": {
            "stage_a": len(stage_a_ids),
            "stage_b_v1": len(stage_b_v1_ids),
            "combined": len(used_ids),
            "combined_sha256": hash_document_ids(used_ids),
        },
        "unused_general": {
            "documents": total_unused_documents,
            "exact_tokens": total_unused_tokens,
            "document_id_sha256": hash_document_ids(all_unused_ids),
            "sources": source_reports,
        },
        "rehearsal": {
            "stage_a_replay_tokens_required": replay_tokens_required,
            "unused_general_is_sufficient": replay_tokens_required == 0,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Unused general documents: {total_unused_documents:,}")
    print(f"Unused general tokens: {total_unused_tokens:,}")
    print(f"Additional general target: {additional_general_tokens:,}")
    print(f"Stage A replay required: {replay_tokens_required:,}")
    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
