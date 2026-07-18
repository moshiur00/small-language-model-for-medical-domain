"""Final corpus assembly from validated dataset files."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_slm.data.assembly.statistics import (
    CorpusStatisticsAccumulator,
    calculate_file_sha256,
)
from medical_slm.data.jsonl import (
    read_jsonl,
    write_jsonl,
)


LOGGER = logging.getLogger(__name__)

VALID_REVIEW_POLICIES = {
    "include",
    "exclude",
}

VALID_OUTPUT_SPLITS = {
    "train",
    "validation",
    "test",
}


def validate_corpus_assembly_config(
    config: Mapping[str, Any],
) -> None:
    """Validate final corpus assembly configuration."""
    required_fields = {
        "output_directory",
        "inputs",
    }

    missing_fields = (
        required_fields - config.keys()
    )

    if missing_fields:
        raise ValueError(
            "Corpus assembly configuration is missing "
            f"fields: {', '.join(sorted(missing_fields))}"
        )

    review_policy = str(
        config.get(
            "review_policy",
            "include",
        )
    )

    if review_policy not in VALID_REVIEW_POLICIES:
        raise ValueError(
            "review_policy must be one of "
            f"{sorted(VALID_REVIEW_POLICIES)}."
        )

    inputs = config["inputs"]

    if (
        not isinstance(inputs, Sequence)
        or isinstance(inputs, str)
        or not inputs
    ):
        raise ValueError(
            "inputs must be a non-empty sequence."
        )

    required_input_fields = {
        "dataset",
        "source_split",
        "output_split",
        "input_path",
    }

    seen_entries: set[
        tuple[str, str, str]
    ] = set()

    for index, entry in enumerate(inputs):
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Input entry {index} must be a mapping."
            )

        missing = (
            required_input_fields - entry.keys()
        )

        if missing:
            raise ValueError(
                f"Input entry {index} is missing fields: "
                f"{', '.join(sorted(missing))}"
            )

        output_split = str(
            entry["output_split"]
        )

        if output_split not in VALID_OUTPUT_SPLITS:
            raise ValueError(
                f"Unsupported output split "
                f"{output_split!r}."
            )

        key = (
            str(entry["dataset"]),
            str(entry["source_split"]),
            output_split,
        )

        if key in seen_entries:
            raise ValueError(
                "Duplicate corpus input entry: "
                f"{key[0]}/{key[1]} -> {key[2]}"
            )

        seen_entries.add(key)

    max_documents = config.get(
        "max_documents_per_input"
    )

    if max_documents is not None:
        if int(max_documents) <= 0:
            raise ValueError(
                "max_documents_per_input must be "
                "greater than zero or null."
            )

    estimated_characters_per_token = float(
        config.get(
            "estimated_characters_per_token",
            4.0,
        )
    )

    if estimated_characters_per_token <= 0:
        raise ValueError(
            "estimated_characters_per_token must "
            "be greater than zero."
        )

    tokenizer_split = str(
        config.get(
            "tokenizer_split",
            "train",
        )
    )

    if tokenizer_split not in VALID_OUTPUT_SPLITS:
        raise ValueError(
            "tokenizer_split must be train, "
            "validation, or test."
        )

    separator = config.get(
        "tokenizer_document_separator",
        "\n",
    )

    if not isinstance(separator, str):
        raise TypeError(
            "tokenizer_document_separator "
            "must be a string."
        )


def get_nested_decision(
    record: Mapping[str, Any],
    metadata_key: str,
) -> str | None:
    """Read one upstream decision from record metadata."""
    metadata = record.get("metadata")

    if not isinstance(metadata, Mapping):
        return None

    stage_metadata = metadata.get(
        metadata_key
    )

    if not isinstance(stage_metadata, Mapping):
        return None

    decision = stage_metadata.get(
        "decision"
    )

    if not isinstance(decision, str):
        return None

    normalized = decision.strip().casefold()

    return normalized or None


def evaluate_record_inclusion(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """
    Determine whether a record can enter the final corpus.

    Returns:
        ``(include_record, exclusion_reasons)``.
    """
    reasons: list[str] = []

    text = record.get("text")

    if not isinstance(text, str):
        reasons.append(
            "invalid_text_type"
        )
    elif not text.strip():
        reasons.append(
            "empty_text"
        )

    record_id = record.get("id")

    if (
        not isinstance(record_id, str)
        or not record_id.strip()
    ):
        reasons.append(
            "missing_document_id"
        )

    quality_decision = get_nested_decision(
        record,
        "quality",
    )

    license_decision = get_nested_decision(
        record,
        "license_validation",
    )

    require_quality = bool(
        config.get(
            "require_quality_metadata",
            True,
        )
    )

    require_license = bool(
        config.get(
            "require_license_metadata",
            True,
        )
    )

    if (
        require_quality
        and quality_decision is None
    ):
        reasons.append(
            "missing_quality_decision"
        )

    if (
        require_license
        and license_decision is None
    ):
        reasons.append(
            "missing_license_decision"
        )

    accepted_quality = {
        str(value).casefold()
        for value in config.get(
            "accepted_quality_decisions",
            ["pass", "review"],
        )
    }

    accepted_license = {
        str(value).casefold()
        for value in config.get(
            "accepted_license_decisions",
            ["pass", "review"],
        )
    }

    if (
        quality_decision is not None
        and quality_decision not in accepted_quality
    ):
        reasons.append(
            "quality_decision_not_accepted"
        )

    if (
        license_decision is not None
        and license_decision not in accepted_license
    ):
        reasons.append(
            "license_decision_not_accepted"
        )

    review_policy = str(
        config.get(
            "review_policy",
            "include",
        )
    )

    if review_policy == "exclude":
        if quality_decision == "review":
            reasons.append(
                "quality_review_excluded"
            )

        if license_decision == "review":
            reasons.append(
                "license_review_excluded"
            )

    return not reasons, reasons


def add_assembly_metadata(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    source_split: str,
    output_split: str,
    input_path: Path,
) -> dict[str, Any]:
    """Return a copied record with assembly provenance metadata."""
    output_record = dict(record)

    existing_metadata = record.get(
        "metadata"
    )

    metadata = (
        dict(existing_metadata)
        if isinstance(existing_metadata, Mapping)
        else {}
    )

    metadata["corpus_assembly"] = {
        "method": "validated_corpus_assembly_v1",
        "dataset": dataset_name,
        "source_split": source_split,
        "output_split": output_split,
        "input_file": str(input_path),
    }

    output_record["metadata"] = metadata

    return output_record


def create_exclusion_record(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    source_split: str,
    output_split: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """Create a compact exclusion-report record."""
    text = record.get("text")

    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "dataset": dataset_name,
        "source_split": source_split,
        "output_split": output_split,
        "reasons": list(reasons),
        "quality_decision": get_nested_decision(
            record,
            "quality",
        ),
        "license_decision": get_nested_decision(
            record,
            "license_validation",
        ),
        "text_preview": (
            text[:300]
            if isinstance(text, str)
            else None
        ),
    }


def iter_assembled_records(
    entry: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    excluded_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield accepted records from one configured input file."""
    dataset_name = str(
        entry["dataset"]
    )
    source_split = str(
        entry["source_split"]
    )
    output_split = str(
        entry["output_split"]
    )
    input_path = Path(
        entry["input_path"]
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Corpus input file does not exist: {input_path}"
        )

    max_documents = config.get(
        "max_documents_per_input"
    )

    processed_from_input = 0

    for record in read_jsonl(
        input_path
    ):
        if (
            max_documents is not None
            and processed_from_input
            >= int(max_documents)
        ):
            break

        processed_from_input += 1

        statistics[
            "input_documents"
        ] += 1

        include_record, reasons = (
            evaluate_record_inclusion(
                record,
                config=config,
            )
        )

        if not include_record:
            statistics[
                "excluded_documents"
            ] += 1

            for reason in reasons:
                reason_counts = statistics[
                    "exclusion_reason_counts"
                ]

                reason_counts[reason] = (
                    reason_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

            excluded_records.append(
                create_exclusion_record(
                    record,
                    dataset_name=dataset_name,
                    source_split=source_split,
                    output_split=output_split,
                    reasons=reasons,
                )
            )
            continue

        statistics[
            "output_documents"
        ] += 1

        if bool(
            config.get(
                "store_assembly_metadata",
                True,
            )
        ):
            yield add_assembly_metadata(
                record,
                dataset_name=dataset_name,
                source_split=source_split,
                output_split=output_split,
                input_path=input_path,
            )
        else:
            yield dict(record)


def normalize_tokenizer_text(
    text: str,
    *,
    flatten: bool,
) -> str:
    """Prepare one document for the tokenizer text corpus."""
    if flatten:
        return " ".join(
            text.split()
        ).strip()

    return text.strip()


def write_tokenizer_corpus(
    records_path: Path,
    output_path: Path,
    *,
    flatten_documents: bool,
    document_separator: str,
) -> dict[str, int]:
    """Create tokenizer training text from one assembled split."""
    if not records_path.exists():
        raise FileNotFoundError(
            "Tokenizer source split does not exist: "
            f"{records_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_count = 0
    character_count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        for record in read_jsonl(
            records_path
        ):
            text = record.get("text")

            if not isinstance(text, str):
                continue

            prepared_text = (
                normalize_tokenizer_text(
                    text,
                    flatten=flatten_documents,
                )
            )

            if not prepared_text:
                continue

            if document_count > 0:
                output_file.write(
                    document_separator
                )

            output_file.write(
                prepared_text
            )

            document_count += 1
            character_count += len(
                prepared_text
            )

    return {
        "document_count": document_count,
        "character_count": character_count,
    }


def build_final_corpus(
    *,
    output_directory: Path,
    inputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build final train, validation, test and tokenizer artifacts."""
    validate_corpus_assembly_config(
        config
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped_inputs: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for entry in inputs:
        grouped_inputs[
            str(entry["output_split"])
        ].append(entry)

    excluded_records: list[
        dict[str, Any]
    ] = []

    split_reports: dict[
        str,
        dict[str, Any]
    ] = {}

    split_statistics: dict[
        str,
        dict[str, Any]
    ] = {}

    for output_split in (
        "train",
        "validation",
        "test",
    ):
        split_entries = grouped_inputs.get(
            output_split,
            [],
        )

        output_path = (
            output_directory
            / f"{output_split}.jsonl"
        )

        assembly_statistics: dict[str, Any] = {
            "split": output_split,
            "output_file": str(output_path),
            "input_documents": 0,
            "output_documents": 0,
            "excluded_documents": 0,
            "exclusion_reason_counts": {},
            "inputs": [
                {
                    "dataset": str(
                        entry["dataset"]
                    ),
                    "source_split": str(
                        entry["source_split"]
                    ),
                    "input_path": str(
                        entry["input_path"]
                    ),
                }
                for entry in split_entries
            ],
        }

        corpus_statistics = (
            CorpusStatisticsAccumulator(
                estimated_characters_per_token=float(
                    config.get(
                        "estimated_characters_per_token",
                        4.0,
                    )
                )
            )
        )

        def records() -> Iterator[
            dict[str, Any]
        ]:
            for entry in split_entries:
                dataset_name = str(
                    entry["dataset"]
                )

                for record in (
                    iter_assembled_records(
                        entry,
                        config=config,
                        statistics=(
                            assembly_statistics
                        ),
                        excluded_records=(
                            excluded_records
                        ),
                    )
                ):
                    corpus_statistics.add_record(
                        record,
                        dataset_name=dataset_name,
                    )

                    yield record

        written_count = write_jsonl(
            records(),
            output_path,
        )

        if (
            written_count
            != assembly_statistics[
                "output_documents"
            ]
        ):
            raise RuntimeError(
                "Written-document count does not match "
                f"assembly statistics for {output_split}."
            )

        split_reports[
            output_split
        ] = assembly_statistics

        split_statistics[
            output_split
        ] = corpus_statistics.to_dict()

        LOGGER.info(
            "Assembled %s: input=%d output=%d excluded=%d",
            output_split,
            assembly_statistics[
                "input_documents"
            ],
            assembly_statistics[
                "output_documents"
            ],
            assembly_statistics[
                "excluded_documents"
            ],
        )

    excluded_path = (
        output_directory
        / "corpus_excluded_documents.jsonl"
    )

    write_jsonl(
        excluded_records,
        excluded_path,
    )

    tokenizer_split = str(
        config.get(
            "tokenizer_split",
            "train",
        )
    )

    tokenizer_source_path = (
        output_directory
        / f"{tokenizer_split}.jsonl"
    )

    tokenizer_filename = str(
        config.get(
            "tokenizer_corpus_filename",
            "tokenizer_corpus.txt",
        )
    )

    tokenizer_path = (
        output_directory
        / tokenizer_filename
    )

    tokenizer_statistics = (
        write_tokenizer_corpus(
            tokenizer_source_path,
            tokenizer_path,
            flatten_documents=bool(
                config.get(
                    "flatten_tokenizer_documents",
                    True,
                )
            ),
            document_separator=str(
                config.get(
                    "tokenizer_document_separator",
                    "\n",
                )
            ),
        )
    )

    total_statistics = {
        "document_count": sum(
            values["document_count"]
            for values in split_statistics.values()
        ),
        "character_count": sum(
            values["character_count"]
            for values in split_statistics.values()
        ),
        "word_count": sum(
            values["word_count"]
            for values in split_statistics.values()
        ),
        "estimated_token_count": sum(
            values["estimated_token_count"]
            for values in split_statistics.values()
        ),
    }

    statistics_document = {
        "generated_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "total": total_statistics,
        "splits": split_statistics,
        "tokenizer_corpus": {
            "source_split": tokenizer_split,
            "path": str(tokenizer_path),
            **tokenizer_statistics,
        },
    }

    statistics_path = (
        output_directory
        / "corpus_statistics.json"
    )

    with statistics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics_document,
            file,
            indent=2,
            ensure_ascii=False,
        )

    artifact_paths = {
        "train": (
            output_directory
            / "train.jsonl"
        ),
        "validation": (
            output_directory
            / "validation.jsonl"
        ),
        "test": (
            output_directory
            / "test.jsonl"
        ),
        "tokenizer_corpus": tokenizer_path,
        "excluded_documents": excluded_path,
        "statistics": statistics_path,
    }

    artifact_manifest = {
        artifact_name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": calculate_file_sha256(
                path
            ),
        }
        for artifact_name, path in artifact_paths.items()
    }

    manifest = {
        "generated_at_utc": (
            datetime.now(UTC).isoformat()
        ),
        "pipeline_stage": (
            "final_corpus_assembly"
        ),
        "input_stage": (
            "license_validated"
        ),
        "review_policy": str(
            config.get(
                "review_policy",
                "include",
            )
        ),
        "max_documents_per_input": (
            config.get(
                "max_documents_per_input"
            )
        ),
        "split_reports": split_reports,
        "artifacts": artifact_manifest,
    }

    manifest_path = (
        output_directory
        / "corpus_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Add the manifest's own file information only after writing it.
    manifest["manifest_file"] = {
        "path": str(manifest_path),
        "size_bytes": (
            manifest_path.stat().st_size
        ),
        "sha256": calculate_file_sha256(
            manifest_path
        ),
    }

    # Rewrite with the manifest-file metadata included.
    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return manifest