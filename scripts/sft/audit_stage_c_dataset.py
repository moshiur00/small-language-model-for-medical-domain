"""Audit Stage C SFT near-duplicate leakage and source-license metadata."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from datasketch import MinHashLSH

from medical_slm.data.deduplication.minhash import (
    calculate_jaccard_similarity,
    canonicalize_for_near_deduplication,
    create_minhash,
    create_word_shingles,
    tokenize_words,
)
from medical_slm.data.tokenization.manifest import calculate_sha256


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/tokenized/sft_stage_c_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/stage_c/stage_c_data_audit.json"),
    )
    parser.add_argument("--shingle-size", type=int, default=3)
    parser.add_argument("--num-permutations", type=int, default=128)
    parser.add_argument("--lsh-threshold", type=float, default=0.75)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Stage C manifest must be a JSON object.")
    records: list[dict[str, Any]] = []
    for split in manifest["splits"]:
        path = root / split / "structured.jsonl"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            records.append({**value, "split": split})
    if len(records) != int(manifest["accepted_examples"]):
        raise ValueError("Structured record total does not match the Stage C manifest.")
    return records, manifest


def prompt_shingles(record: Mapping[str, Any], shingle_size: int) -> set[str]:
    text = "\n".join(
        (
            str(record.get("instruction", "")),
            str(record.get("context_type", "none")),
            str(record.get("context", "")),
        )
    )
    canonical = canonicalize_for_near_deduplication(text)
    words = tokenize_words(canonical)
    if not words:
        raise ValueError(f"SFT prompt contains no comparison words: {record.get('id')}")
    return create_word_shingles(words, shingle_size=min(shingle_size, len(words)))


class DisjointSet:
    """Minimal deterministic union-find for near-duplicate cluster counts."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def find_near_duplicate_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    shingle_size: int = 3,
    num_permutations: int = 128,
    lsh_threshold: float = 0.75,
    similarity_threshold: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    if shingle_size <= 0:
        raise ValueError("shingle_size must be greater than zero.")
    if not 0.0 < lsh_threshold <= similarity_threshold <= 1.0:
        raise ValueError(
            "Expected 0 < lsh_threshold <= similarity_threshold <= 1."
        )
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_permutations)
    indexed_shingles: dict[str, set[str]] = {}
    indexed_records: dict[str, Mapping[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    clusters = DisjointSet()
    candidate_matches = 0

    ordered = sorted(
        records,
        key=lambda record: (
            str(record.get("split")),
            str(record.get("source")),
            str(record.get("id")),
        ),
    )
    for record in ordered:
        record_id = str(record.get("id", ""))
        split = str(record.get("split", ""))
        key = f"{split}:{record_id}"
        if not record_id or key in indexed_records:
            raise ValueError(f"Missing or duplicate audit record key: {key}")
        shingles = prompt_shingles(record, shingle_size)
        signature = create_minhash(
            shingles,
            num_permutations=num_permutations,
            random_seed=seed,
        )
        candidates = sorted(lsh.query(signature))
        candidate_matches += len(candidates)
        clusters.add(key)
        for candidate_key in candidates:
            candidate = indexed_records[candidate_key]
            if record.get("split_group_sha256") == candidate.get(
                "split_group_sha256"
            ):
                continue
            similarity = calculate_jaccard_similarity(
                shingles,
                indexed_shingles[candidate_key],
            )
            if similarity < similarity_threshold:
                continue
            clusters.union(key, candidate_key)
            pairs.append(
                {
                    "left_id": str(candidate.get("id")),
                    "left_source": str(candidate.get("source")),
                    "left_split": str(candidate.get("split")),
                    "left_prompt_group_sha256": str(
                        candidate.get("split_group_sha256")
                    ),
                    "right_id": record_id,
                    "right_source": str(record.get("source")),
                    "right_split": split,
                    "right_prompt_group_sha256": str(
                        record.get("split_group_sha256")
                    ),
                    "exact_jaccard_similarity": round(similarity, 6),
                }
            )
        lsh.insert(key, signature)
        indexed_shingles[key] = shingles
        indexed_records[key] = record

    cluster_members: dict[str, list[str]] = defaultdict(list)
    for key in clusters.parent:
        cluster_members[clusters.find(key)].append(key)
    duplicate_clusters = [
        members for members in cluster_members.values() if len(members) > 1
    ]
    cross_split_clusters = 0
    for members in duplicate_clusters:
        splits = {str(indexed_records[key].get("split")) for key in members}
        cross_split_clusters += int(len(splits) > 1)

    pairs.sort(
        key=lambda pair: (
            pair["left_split"],
            pair["left_id"],
            pair["right_split"],
            pair["right_id"],
        )
    )
    return {
        "configuration": {
            "representation": "normalized_instruction_context_word_shingles",
            "shingle_size": shingle_size,
            "num_permutations": num_permutations,
            "lsh_threshold": lsh_threshold,
            "similarity_threshold": similarity_threshold,
            "seed": seed,
        },
        "records": len(records),
        "candidate_matches": candidate_matches,
        "near_duplicate_pairs": len(pairs),
        "cross_source_pairs": sum(
            pair["left_source"] != pair["right_source"] for pair in pairs
        ),
        "cross_split_pairs": sum(
            pair["left_split"] != pair["right_split"] for pair in pairs
        ),
        "near_duplicate_clusters": len(duplicate_clusters),
        "cross_split_clusters": cross_split_clusters,
        "pairs": pairs,
    }


def summarize_licenses(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    invalid_decisions = 0
    review_examples = 0
    for record in records:
        source = str(record.get("source", "unknown"))
        summary = sources.setdefault(
            source,
            {
                "examples": 0,
                "licenses": set(),
                "decisions": defaultdict(int),
                "commercial_use": set(),
                "redistribution": set(),
            },
        )
        summary["examples"] += 1
        summary["licenses"].add(str(record.get("license", "unknown")))
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        license_metadata = metadata.get("license_validation")
        if not isinstance(license_metadata, Mapping):
            license_metadata = {}
        decision = str(license_metadata.get("decision", "unknown"))
        summary["decisions"][decision] += 1
        invalid_decisions += int(decision not in {"pass", "review"})
        review_examples += int(decision == "review")
        obligations = license_metadata.get("obligations")
        if not isinstance(obligations, Mapping):
            obligations = {}
        summary["commercial_use"].add(obligations.get("commercial_use"))
        summary["redistribution"].add(obligations.get("redistribution"))

    serialized = {}
    for source, summary in sorted(sources.items()):
        serialized[source] = {
            "examples": summary["examples"],
            "licenses": sorted(summary["licenses"]),
            "decisions": dict(sorted(summary["decisions"].items())),
            "commercial_use": sorted(
                summary["commercial_use"], key=lambda value: str(value)
            ),
            "redistribution": sorted(
                summary["redistribution"], key=lambda value: str(value)
            ),
            "public_model_release_status": (
                "requires_manual_review"
                if summary["decisions"].get("review", 0)
                else "not_assessed_by_dataset_license_gate"
            ),
        }
    return {
        "sources": serialized,
        "invalid_license_decision_examples": invalid_decisions,
        "manual_review_examples": review_examples,
        "internal_research_training_status": (
            "passed" if invalid_decisions == 0 else "blocked"
        ),
        "public_model_release_status": (
            "blocked_pending_manual_review"
            if review_examples or invalid_decisions
            else "not_assessed_by_dataset_license_gate"
        ),
        "interpretation": (
            "Dataset-license metadata can gate internal corpus use, but it does not "
            "by itself determine whether trained model weights may be distributed."
        ),
    }


def audit_stage_c_dataset(
    *,
    root: Path,
    output: Path,
    shingle_size: int = 3,
    num_permutations: int = 128,
    lsh_threshold: float = 0.75,
    similarity_threshold: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    records, manifest = read_records(root)
    near_duplicates = find_near_duplicate_pairs(
        records,
        shingle_size=shingle_size,
        num_permutations=num_permutations,
        lsh_threshold=lsh_threshold,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    licenses = summarize_licenses(records)
    near_duplicate_status = (
        "passed" if near_duplicates["cross_split_pairs"] == 0 else "blocked"
    )
    report = {
        "format_version": 1,
        "stage": "supervised_instruction_finetuning_stage_c_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(root),
            "manifest_sha256": calculate_sha256(root / "manifest.json"),
            "input_sha256": manifest["input_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "accepted_examples": manifest["accepted_examples"],
            "splits": {
                name: int(value["examples"])
                for name, value in manifest["splits"].items()
            },
        },
        "near_duplicate_audit": {
            **near_duplicates,
            "status": near_duplicate_status,
        },
        "license_audit": licenses,
        "training_readiness": {
            "near_duplicate_gate": near_duplicate_status,
            "internal_research_license_gate": licenses[
                "internal_research_training_status"
            ],
            "public_model_release_gate": licenses["public_model_release_status"],
        },
        "status": (
            "passed_for_internal_research"
            if near_duplicate_status == "passed"
            and licenses["internal_research_training_status"] == "passed"
            else "blocked"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    arguments = parse_arguments()
    report = audit_stage_c_dataset(
        root=arguments.root,
        output=arguments.output,
        shingle_size=arguments.shingle_size,
        num_permutations=arguments.num_permutations,
        lsh_threshold=arguments.lsh_threshold,
        similarity_threshold=arguments.similarity_threshold,
        seed=arguments.seed,
    )
    near = report["near_duplicate_audit"]
    license_audit = report["license_audit"]
    print("Stage C data audit:", report["status"])
    print("Near-duplicate pairs:", near["near_duplicate_pairs"])
    print("Cross-source pairs:", near["cross_source_pairs"])
    print("Cross-split pairs:", near["cross_split_pairs"])
    print("Manual-review examples:", license_audit["manual_review_examples"])
    print("Public release:", license_audit["public_model_release_status"])
    print("Report:", arguments.output)
    if report["status"] == "blocked":
        raise SystemExit("Stage C data audit is blocked; inspect the report.")


if __name__ == "__main__":
    main()
