"""Assemble independent token-budgeted pretraining and balanced SFT corpora."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from medical_slm.data.jsonl import read_jsonl


def estimate_tokens(text: str, characters_per_token: float) -> int:
    """Estimate tokens before the project tokenizer exists."""
    return max(1, round(len(text) / characters_per_token))


def _write_record(file: Any, record: Mapping[str, Any]) -> None:
    file.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")))
    file.write("\n")


def assemble_token_budget_phase(
    *,
    phase_name: str,
    phase_config: Mapping[str, Any],
    input_directory: Path,
    output_directory: Path,
    characters_per_token: float,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Fill each source quota from toxicity-audited training records."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "train.jsonl"
    text_path = output_directory / "corpus.txt"
    source_reports: dict[str, dict[str, Any]] = {}
    total_documents = 0
    total_tokens = 0

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
            for record in read_jsonl(input_path):
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
            }
            total_documents += documents
            total_tokens += tokens

    return {
        "phase": phase_name,
        "target_tokens": int(phase_config["target_tokens"]),
        "token_count_method": "exact_tokenizer" if token_counter else "characters_per_token",
        "tokens": total_tokens,
        "estimated_tokens": total_tokens if token_counter is None else None,
        "exact_tokens": total_tokens if token_counter is not None else None,
        "documents": total_documents,
        "sources": source_reports,
        "artifacts": {"records": str(output_path), "text": str(text_path)},
    }


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
            reports[phase_name] = assemble_token_budget_phase(
                phase_name=phase_name,
                phase_config=phase_config,
                input_directory=input_directory,
                output_directory=phase_output,
                characters_per_token=characters_per_token,
                token_counter=token_counter,
            )
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
