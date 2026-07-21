"""Assemble independent token-budgeted pretraining and balanced SFT corpora."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import AbstractSet, Any

import yaml

from medical_slm.data.jsonl import read_jsonl
from medical_slm.data.tokenization.manifest import calculate_sha256


def estimate_tokens(text: str, characters_per_token: float) -> int:
    """Estimate tokens before the project tokenizer exists."""
    return max(1, round(len(text) / characters_per_token))


def _write_record(file: Any, record: Mapping[str, Any]) -> None:
    file.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")))
    file.write("\n")


def document_id(record: Mapping[str, Any]) -> str:
    """Return one validated stable document identifier."""
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every corpus record must contain a non-empty string id.")
    return value


def load_document_ids(path: Path) -> set[str]:
    """Load unique document identifiers and reject duplicates."""
    identifiers: set[str] = set()
    for record in read_jsonl(path):
        identifier = document_id(record)
        if identifier in identifiers:
            raise ValueError(f"Duplicate document id in {path}: {identifier}")
        identifiers.add(identifier)
    return identifiers


def hash_document_ids(identifiers: AbstractSet[str]) -> str:
    """Hash a set of document IDs independently of record order."""
    digest = hashlib.sha256()
    for identifier in sorted(identifiers):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assemble_token_budget_phase(
    *,
    phase_name: str,
    phase_config: Mapping[str, Any],
    input_directory: Path,
    output_directory: Path,
    characters_per_token: float,
    token_counter: Callable[[str], int] | None = None,
    excluded_document_ids: AbstractSet[str] = frozenset(),
) -> dict[str, Any]:
    """Fill each source quota from toxicity-audited training records."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "train.jsonl"
    text_path = output_directory / "corpus.txt"
    source_reports: dict[str, dict[str, Any]] = {}
    total_documents = 0
    total_tokens = 0
    selected_document_ids: set[str] = set()
    total_excluded_documents = 0

    with output_path.open("w", encoding="utf-8") as output_file, text_path.open(
        "w", encoding="utf-8"
    ) as text_file:
        for source, allocation in phase_config["sources"].items():
            target = int(allocation["tokens"])
            input_path = input_directory / source / "train.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing toxicity-audited source: {input_path}")
            documents = 0
            tokens = 0
            excluded_documents = 0
            for record in read_jsonl(input_path):
                identifier = document_id(record)
                if identifier in excluded_document_ids:
                    excluded_documents += 1
                    continue
                if identifier in selected_document_ids:
                    raise ValueError(
                        f"Duplicate document id selected for {phase_name}: {identifier}"
                    )
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                record_tokens = (
                    token_counter(text)
                    if token_counter is not None
                    else estimate_tokens(text, characters_per_token)
                )
                if tokens and tokens + record_tokens > target:
                    break
                _write_record(output_file, record)
                text_file.write(" ".join(text.split()))
                text_file.write("\n")
                documents += 1
                selected_document_ids.add(identifier)
                tokens += record_tokens
                if tokens >= target:
                    break
            source_reports[source] = {
                "target_tokens": target,
                "tokens": tokens,
                "estimated_tokens": tokens if token_counter is None else None,
                "exact_tokens": tokens if token_counter is not None else None,
                "documents": documents,
                "domain": allocation.get("domain"),
                "exhausted_before_target": tokens < target,
                "excluded_documents": excluded_documents,
            }
            total_documents += documents
            total_tokens += tokens
            total_excluded_documents += excluded_documents

    return {
        "phase": phase_name,
        "target_tokens": int(phase_config["target_tokens"]),
        "token_count_method": "exact_tokenizer" if token_counter else "characters_per_token",
        "tokens": total_tokens,
        "estimated_tokens": total_tokens if token_counter is None else None,
        "exact_tokens": total_tokens if token_counter is not None else None,
        "documents": total_documents,
        "document_id_sha256": hash_document_ids(selected_document_ids),
        "excluded_documents": total_excluded_documents,
        "excluded_document_id_count": len(excluded_document_ids),
        "sources": source_reports,
        "artifacts": {"records": str(output_path), "text": str(text_path)},
    }


def assemble_disjoint_evaluation_corpora(
    *,
    evaluation_config: Mapping[str, Any],
    input_directory: Path,
    output_directory: Path,
    excluded_document_ids: AbstractSet[str],
    characters_per_token: float,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Build mutually disjoint evaluation splits from unused source documents."""
    raw_splits = evaluation_config.get("splits")
    if not isinstance(raw_splits, Mapping) or not raw_splits:
        raise TypeError("Medical evaluation configuration requires non-empty splits.")

    source_names: list[str] = []
    for split_config in raw_splits.values():
        if not isinstance(split_config, Mapping):
            raise TypeError("Each medical evaluation split must be a mapping.")
        sources = split_config.get("sources")
        if not isinstance(sources, Mapping) or not sources:
            raise TypeError("Each medical evaluation split requires sources.")
        for source in sources:
            if source not in source_names:
                source_names.append(str(source))

    iterators: dict[str, Iterator[dict[str, Any]]] = {}
    for source in source_names:
        path = input_directory / source / "train.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation source corpus: {path}")
        iterators[source] = iter(read_jsonl(path))

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Medical evaluation output directory is not empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    selected_document_ids: set[str] = set()
    split_reports: dict[str, Any] = {}

    for split_name, split_config in raw_splits.items():
        output_path = output_directory / f"{split_name}.jsonl"
        split_ids: set[str] = set()
        source_reports: dict[str, Any] = {}
        split_tokens = 0
        split_documents = 0

        with output_path.open("w", encoding="utf-8") as output_file:
            for source, allocation in split_config["sources"].items():
                if not isinstance(allocation, Mapping) or "tokens" not in allocation:
                    raise TypeError(
                        f"Medical evaluation allocation for {split_name}/{source} "
                        "requires a token target."
                    )
                target_tokens = int(allocation["tokens"])
                if target_tokens <= 0:
                    raise ValueError("Medical evaluation token targets must be positive.")

                source_tokens = 0
                source_documents = 0
                excluded_documents = 0
                while source_tokens < target_tokens:
                    try:
                        record = next(iterators[str(source)])
                    except StopIteration as error:
                        raise ValueError(
                            f"Source {source} was exhausted while building {split_name}."
                        ) from error
                    identifier = document_id(record)
                    if (
                        identifier in excluded_document_ids
                        or identifier in selected_document_ids
                    ):
                        excluded_documents += 1
                        continue
                    text = record.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    record_tokens = (
                        token_counter(text)
                        if token_counter is not None
                        else estimate_tokens(text, characters_per_token)
                    )
                    _write_record(output_file, record)
                    selected_document_ids.add(identifier)
                    split_ids.add(identifier)
                    source_documents += 1
                    source_tokens += record_tokens

                source_reports[str(source)] = {
                    "target_tokens": target_tokens,
                    "tokens": source_tokens,
                    "documents": source_documents,
                    "excluded_documents_scanned": excluded_documents,
                    "domain": allocation.get("domain", "medical"),
                }
                split_tokens += source_tokens
                split_documents += source_documents

        split_reports[str(split_name)] = {
            "target_tokens": sum(
                int(allocation["tokens"])
                for allocation in split_config["sources"].values()
            ),
            "tokens": split_tokens,
            "documents": split_documents,
            "document_id_sha256": hash_document_ids(split_ids),
            "sources": source_reports,
            "artifact": {
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": calculate_sha256(output_path),
            },
        }

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "format_version": 1,
        "method": "source_stratified_document_disjoint_evaluation_v1",
        "token_count_method": "exact_tokenizer" if token_counter else "characters_per_token",
        "excluded_document_id_count": len(excluded_document_ids),
        "excluded_document_id_sha256": hash_document_ids(excluded_document_ids),
        "selected_document_id_count": len(selected_document_ids),
        "selected_document_id_sha256": hash_document_ids(selected_document_ids),
        "splits": split_reports,
    }
    manifest_path = output_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return manifest


def _count_records(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def assemble_balanced_sft(
    *,
    phase_name: str,
    phase_config: Mapping[str, Any],
    input_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create an equal capped round-robin mixture of SFT sources."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        source: input_directory / source / "train.jsonl"
        for source in phase_config["sources"]
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing toxicity-audited SFT source: {path}")
    counts = {source: _count_records(path) for source, path in paths.items()}
    cap = min(counts.values()) if counts else 0
    iterators: dict[str, Iterator[dict[str, Any]]] = {
        source: iter(read_jsonl(path)) for source, path in paths.items()
    }
    written = Counter()
    output_path = output_directory / "train.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for _ in range(cap):
            for source, iterator in iterators.items():
                record = next(iterator)
                _write_record(output_file, record)
                written[source] += 1
    return {
        "phase": phase_name,
        "balance_by": phase_config.get("balance_by", "examples"),
        "sampling": phase_config.get("sampling", "capped_round_robin"),
        "per_source_cap": cap,
        "documents": sum(written.values()),
        "available_documents": counts,
        "written_documents": dict(written),
        "artifacts": {"records": str(output_path)},
    }


def build_phase_corpora(
    *,
    corpora_config_path: Path,
    input_directory: Path = Path("datasets/interim/toxicity_audited"),
    output_directory: Path = Path("datasets/processed/corpora"),
    token_counter: Callable[[str], int] | None = None,
    tokenizer_path: Path | None = None,
    selected_phases: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build every configured phase and write one aggregate manifest."""
    with corpora_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    corpora = config["corpora"]
    characters_per_token = float(config.get("estimated_characters_per_token", 4.0))
    if selected_phases is not None:
        unknown_phases = set(selected_phases) - set(corpora)
        if unknown_phases:
            raise ValueError(
                "Unknown corpus phases: " + ", ".join(sorted(unknown_phases))
            )
    manifest_path = output_directory / "manifest.json"
    reports: dict[str, Any] = {}
    if selected_phases is not None and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            previous_manifest = json.load(file)
        previous_reports = previous_manifest.get("phases", {})
        if isinstance(previous_reports, Mapping):
            reports.update(previous_reports)
    for phase_name, phase_config in corpora.items():
        if selected_phases is not None and phase_name not in selected_phases:
            continue
        phase_output = output_directory / phase_name
        if "target_tokens" in phase_config:
            excluded_phases = phase_config.get("exclude_phases", [])
            if not isinstance(excluded_phases, Sequence) or isinstance(
                excluded_phases, (str, bytes)
            ):
                raise TypeError(f"exclude_phases for {phase_name} must be a sequence.")
            excluded_document_ids: set[str] = set()
            for excluded_phase in excluded_phases:
                excluded_path = output_directory / str(excluded_phase) / "train.jsonl"
                if not excluded_path.is_file():
                    raise FileNotFoundError(
                        f"Excluded phase corpus is missing: {excluded_path}"
                    )
                excluded_document_ids.update(load_document_ids(excluded_path))
            reports[phase_name] = assemble_token_budget_phase(
                phase_name=phase_name,
                phase_config=phase_config,
                input_directory=input_directory,
                output_directory=phase_output,
                characters_per_token=characters_per_token,
                token_counter=token_counter,
                excluded_document_ids=excluded_document_ids,
            )
            reports[phase_name]["excluded_phases"] = list(excluded_phases)
            reports[phase_name]["input_stage"] = input_directory.name
            reports[phase_name]["input_directory"] = str(input_directory)
        else:
            reports[phase_name] = assemble_balanced_sft(
                phase_name=phase_name,
                phase_config=phase_config,
                input_directory=input_directory,
                output_directory=phase_output,
            )
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_stage": input_directory.name,
        "input_directory": str(input_directory),
        "token_count_method": (
            "mixed_by_phase"
            if selected_phases is not None
            else ("exact_tokenizer" if token_counter else "characters_per_token")
        ),
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "selected_phases": list(selected_phases) if selected_phases is not None else None,
        "estimated_characters_per_token": characters_per_token,
        "phases": reports,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest
