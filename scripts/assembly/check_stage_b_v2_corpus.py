"""Verify Stage B v2 corpus identity, proportions, and overlap contracts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_slm.data.assembly.phases import hash_document_ids, load_document_ids
from medical_slm.data.tokenization import calculate_sha256


PHASE_NAME = "continual_medical_stage_b_v2_70_30"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-corpus",
        type=Path,
        default=Path(f"datasets/processed/corpora/{PHASE_NAME}/train.jsonl"),
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
        "--medical-validation",
        type=Path,
        default=Path("datasets/processed/evaluation_medical/validation.jsonl"),
    )
    parser.add_argument(
        "--medical-test",
        type=Path,
        default=Path("datasets/processed/evaluation_medical/test.jsonl"),
    )
    parser.add_argument(
        "--general-validation",
        type=Path,
        default=Path("datasets/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--general-test",
        type=Path,
        default=Path("datasets/processed/test.jsonl"),
    )
    parser.add_argument(
        "--phase-manifest",
        type=Path,
        default=Path("datasets/processed/corpora/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_b/v2/corpus_verification.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    arguments = parse_arguments()
    v2_ids = load_document_ids(arguments.v2_corpus)
    stage_a_ids = load_document_ids(arguments.stage_a_corpus)
    v1_ids = load_document_ids(arguments.stage_b_v1_corpus)
    evaluation_sets = {
        "medical_validation": load_document_ids(arguments.medical_validation),
        "medical_test": load_document_ids(arguments.medical_test),
        "general_validation": load_document_ids(arguments.general_validation),
        "general_test": load_document_ids(arguments.general_test),
    }

    stage_a_overlap = v2_ids & stage_a_ids
    missing_v1_documents = v1_ids - v2_ids
    evaluation_overlaps = {
        name: len(v2_ids & identifiers)
        for name, identifiers in evaluation_sets.items()
    }
    if stage_a_overlap:
        raise ValueError("Stage B v2 overlaps Stage A.")
    if missing_v1_documents:
        raise ValueError("Stage B v2 does not retain the complete v1 corpus.")
    if any(evaluation_overlaps.values()):
        raise ValueError("Stage B v2 overlaps an evaluation split.")

    manifest = read_json(arguments.phase_manifest)
    phase = manifest["phases"][PHASE_NAME]
    medical_tokens = sum(
        int(source["exact_tokens"])
        for source in phase["sources"].values()
        if source["domain"] == "medical"
    )
    general_tokens = sum(
        int(source["exact_tokens"])
        for source in phase["sources"].values()
        if source["domain"] == "general"
    )
    total_tokens = medical_tokens + general_tokens
    if total_tokens != int(phase["exact_tokens"]):
        raise ValueError("Stage B v2 source token counts do not sum to the phase total.")

    report = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": str(arguments.v2_corpus),
            "size_bytes": arguments.v2_corpus.stat().st_size,
            "sha256": calculate_sha256(arguments.v2_corpus),
            "documents": len(v2_ids),
            "document_id_sha256": hash_document_ids(v2_ids),
            "exact_tokens": total_tokens,
            "medical_tokens": medical_tokens,
            "general_tokens": general_tokens,
            "medical_fraction": medical_tokens / total_tokens,
            "general_fraction": general_tokens / total_tokens,
        },
        "relationships": {
            "stage_a_overlap": len(stage_a_overlap),
            "v1_documents": len(v1_ids),
            "v1_documents_retained": len(v1_ids & v2_ids),
            "new_v2_documents": len(v2_ids - v1_ids),
            "evaluation_overlaps": evaluation_overlaps,
        },
        "status": "passed",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Stage B v2 corpus verification: PASSED")
    print(f"Documents: {len(v2_ids):,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Medical/general: {medical_tokens / total_tokens:.4%}/"
          f"{general_tokens / total_tokens:.4%}")
    print(f"New v2 documents: {len(v2_ids - v1_ids):,}")
    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
