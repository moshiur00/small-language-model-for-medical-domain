"""Global near-duplicate removal using MinHash and exact Jaccard verification."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from medical_slm.data.jsonl import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"\w+(?:['’-]\w+)*", flags=re.UNICODE)

VALID_UNICODE_FORMS = {
    "NFC",
    "NFD",
    "NFKC",
    "NFKD",
}


def canonicalize_for_near_deduplication(
    text: str,
    *,
    unicode_normalization: str = "NFKC",
    normalize_whitespace: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Canonicalize text for near-duplicate comparison.

    This comparison representation does not replace the model-visible text
    stored in the output record.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"text must be a string, received {type(text).__name__}"
        )

    if unicode_normalization not in VALID_UNICODE_FORMS:
        raise ValueError(
            "Unsupported Unicode normalization form "
            f"{unicode_normalization!r}. Expected one of "
            f"{sorted(VALID_UNICODE_FORMS)}."
        )

    canonical_text = unicodedata.normalize(
        unicode_normalization,
        text,
    )

    if lowercase:
        canonical_text = canonical_text.casefold()

    if normalize_whitespace:
        canonical_text = WHITESPACE_PATTERN.sub(
            " ",
            canonical_text,
        )

    return canonical_text.strip()


def tokenize_words(text: str) -> list[str]:
    """Extract comparison tokens from canonicalized text."""
    return WORD_PATTERN.findall(text)


def create_word_shingles(
    words: Sequence[str],
    *,
    shingle_size: int,
) -> set[str]:
    """
    Create a set of contiguous word shingles.

    For a document shorter than the requested shingle size, one shingle
    containing the complete document is returned.
    """
    if shingle_size <= 0:
        raise ValueError("shingle_size must be greater than zero.")

    if not words:
        return set()

    if len(words) < shingle_size:
        return {" ".join(words)}

    return {
        " ".join(words[index : index + shingle_size])
        for index in range(
            len(words) - shingle_size + 1
        )
    }


def calculate_jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:
    """Calculate exact Jaccard similarity between two shingle sets."""
    if not first and not second:
        return 1.0

    if not first or not second:
        return 0.0

    intersection_size = len(first & second)
    union_size = len(first | second)

    return intersection_size / union_size


def create_minhash(
    shingles: set[str],
    *,
    num_permutations: int,
    random_seed: int,
) -> MinHash:
    """Create a MinHash signature from a set of shingles."""
    if num_permutations <= 0:
        raise ValueError(
            "num_permutations must be greater than zero."
        )

    if not shingles:
        raise ValueError(
            "Cannot create a MinHash signature from an empty set."
        )

    signature = MinHash(
        num_perm=num_permutations,
        seed=random_seed,
    )

    for shingle in sorted(shingles):
        signature.update(shingle.encode("utf-8"))

    return signature


def prepare_document(
    record: Mapping[str, Any],
    *,
    unicode_normalization: str,
    normalize_whitespace: bool,
    lowercase: bool,
    shingle_size: int,
) -> tuple[list[str] | None, set[str] | None, str | None]:
    """
    Prepare one record for near-duplicate comparison.

    Returns:
        Word tokens, shingles and an optional rejection reason.
    """
    text = record.get("text")

    if not isinstance(text, str):
        return None, None, "invalid_text_type"

    canonical_text = canonicalize_for_near_deduplication(
        text,
        unicode_normalization=unicode_normalization,
        normalize_whitespace=normalize_whitespace,
        lowercase=lowercase,
    )

    if not canonical_text:
        return None, None, "empty_canonical_text"

    words = tokenize_words(canonical_text)

    if not words:
        return None, None, "no_words"

    shingles = create_word_shingles(
        words,
        shingle_size=shingle_size,
    )

    if not shingles:
        return None, None, "no_shingles"

    return words, shingles, None


def add_near_deduplication_metadata(
    record: Mapping[str, Any],
    *,
    word_count: int,
    shingle_count: int,
    shingle_size: int,
    num_permutations: int,
    indexed: bool,
    store_signature_metadata: bool,
) -> dict[str, Any]:
    """Return a record with near-deduplication metadata."""
    output_record = dict(record)

    existing_metadata = record.get("metadata")
    metadata = (
        dict(existing_metadata)
        if isinstance(existing_metadata, Mapping)
        else {}
    )

    near_metadata: dict[str, Any] = {
        "method": "minhash_lsh_with_exact_jaccard",
        "indexed": indexed,
    }

    if store_signature_metadata:
        near_metadata.update(
            {
                "word_count": word_count,
                "shingle_count": shingle_count,
                "shingle_size": shingle_size,
                "num_permutations": num_permutations,
            }
        )

    metadata["near_deduplication"] = near_metadata
    output_record["metadata"] = metadata

    return output_record


def validate_priority_entries(
    priority: Sequence[Mapping[str, Any]],
) -> None:
    """Validate globally ordered dataset and split entries."""
    if not priority:
        raise ValueError(
            "Near-deduplication priority must contain "
            "at least one entry."
        )

    required_fields = {
        "dataset",
        "split",
        "input_path",
    }

    seen_entries: set[tuple[str, str]] = set()

    for index, entry in enumerate(priority):
        missing_fields = required_fields - entry.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))

            raise ValueError(
                f"Priority entry {index} is missing fields: {missing}"
            )

        key = (
            str(entry["dataset"]),
            str(entry["split"]),
        )

        if key in seen_entries:
            raise ValueError(
                "Near-deduplication priority contains duplicate "
                f"dataset/split entry: {key[0]}/{key[1]}"
            )

        seen_entries.add(key)


def iter_near_unique_records(
    input_path: Path,
    *,
    dataset_name: str,
    split: str,
    lsh: MinHashLSH,
    kept_shingles: dict[str, set[str]],
    kept_records: dict[str, dict[str, Any]],
    indexed_document_ids: deque[str],
    config: Mapping[str, Any],
    statistics: dict[str, Any],
    duplicate_pairs: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield records that are not verified near-duplicates."""
    shingle_size = int(config.get("shingle_size", 5))
    min_words = int(config.get("min_words", 20))
    num_permutations = int(
        config.get("num_permutations", 128)
    )
    random_seed = int(config.get("random_seed", 42))

    unicode_normalization = str(
        config.get("unicode_normalization", "NFKC")
    )
    normalize_whitespace = bool(
        config.get("normalize_whitespace", True)
    )
    lowercase = bool(config.get("lowercase", True))

    similarity_threshold = float(
        config.get("similarity_threshold", 0.90)
    )
    store_signature_metadata = bool(
        config.get("store_signature_metadata", True)
    )
    max_indexed_documents = int(
        config.get("max_indexed_documents", 2000)
    )

    if max_indexed_documents <= 0:
        raise ValueError(
            "max_indexed_documents must be greater than zero."
        )

    for source_index, record in enumerate(
        tqdm(
            read_jsonl(input_path),
            desc=f"Near-deduplicating {dataset_name}/{split}",
            unit="documents",
        )
    ):
        statistics["input_documents"] += 1

        words, shingles, rejection_reason = prepare_document(
            record,
            unicode_normalization=unicode_normalization,
            normalize_whitespace=normalize_whitespace,
            lowercase=lowercase,
            shingle_size=shingle_size,
        )

        if words is None or shingles is None:
            statistics["rejected_documents"] += 1

            rejection_counts = statistics["rejection_counts"]
            rejection_counts[rejection_reason] = (
                rejection_counts.get(rejection_reason, 0) + 1
            )
            continue

        document_id = record.get("id")

        if not isinstance(document_id, str) or not document_id:
            document_id = (
                f"{dataset_name}-{split}-near-{source_index:09d}"
            )

        # Very short documents do not provide enough shingles for reliable
        # near-duplicate comparison. Keep them without adding them to LSH.
        if len(words) < min_words:
            statistics["short_documents_not_indexed"] += 1
            statistics["output_documents"] += 1

            yield add_near_deduplication_metadata(
                record,
                word_count=len(words),
                shingle_count=len(shingles),
                shingle_size=shingle_size,
                num_permutations=num_permutations,
                indexed=False,
                store_signature_metadata=store_signature_metadata,
            )
            continue

        signature = create_minhash(
            shingles,
            num_permutations=num_permutations,
            random_seed=random_seed,
        )

        candidate_ids = lsh.query(signature)
        statistics["candidate_matches"] += len(candidate_ids)

        best_match_id: str | None = None
        best_similarity = 0.0

        for candidate_id in candidate_ids:
            candidate_shingles = kept_shingles[candidate_id]

            similarity = calculate_jaccard_similarity(
                shingles,
                candidate_shingles,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = candidate_id

        if (
            best_match_id is not None
            and best_similarity >= similarity_threshold
        ):
            statistics["near_duplicate_documents"] += 1

            kept_record = kept_records[best_match_id]

            duplicate_pairs.append(
                {
                    "removed_id": document_id,
                    "removed_dataset": dataset_name,
                    "removed_split": split,
                    "kept_id": best_match_id,
                    "kept_dataset": kept_record["dataset"],
                    "kept_split": kept_record["split"],
                    "exact_jaccard_similarity": round(
                        best_similarity,
                        6,
                    ),
                    "shingle_size": shingle_size,
                }
            )
            continue

        lsh.insert(document_id, signature)
        kept_shingles[document_id] = shingles
        kept_records[document_id] = {
            "dataset": dataset_name,
            "split": split,
        }
        indexed_document_ids.append(document_id)

        # MinHashLSH and exact-verification shingles are both sizeable.
        # Keeping a rolling window prevents memory use from growing with the
        # full corpus while still comparing against recent retained records.
        while len(indexed_document_ids) > max_indexed_documents:
            evicted_id = indexed_document_ids.popleft()
            lsh.remove(evicted_id)
            kept_shingles.pop(evicted_id, None)
            kept_records.pop(evicted_id, None)
            statistics["evicted_index_documents"] += 1

        statistics["indexed_documents"] += 1
        statistics["output_documents"] += 1

        yield add_near_deduplication_metadata(
            record,
            word_count=len(words),
            shingle_count=len(shingles),
            shingle_size=shingle_size,
            num_permutations=num_permutations,
            indexed=True,
            store_signature_metadata=store_signature_metadata,
        )


def run_global_near_deduplication(
    *,
    priority: Sequence[Mapping[str, Any]],
    output_directory: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run MinHash-based near-duplicate removal across all datasets."""
    validate_priority_entries(priority)

    num_permutations = int(
        config.get("num_permutations", 128)
    )
    lsh_threshold = float(
        config.get("lsh_threshold", 0.80)
    )
    similarity_threshold = float(
        config.get("similarity_threshold", 0.90)
    )

    if not 0.0 < lsh_threshold <= 1.0:
        raise ValueError(
            "lsh_threshold must be greater than 0 and at most 1."
        )

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be greater than 0 and at most 1."
        )

    if lsh_threshold > similarity_threshold:
        raise ValueError(
            "lsh_threshold should not exceed similarity_threshold, "
            "because qualifying documents may not become candidates."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    lsh = MinHashLSH(
        threshold=lsh_threshold,
        num_perm=num_permutations,
    )

    kept_shingles: dict[str, set[str]] = {}
    kept_records: dict[str, dict[str, Any]] = {}
    indexed_document_ids: deque[str] = deque()
    duplicate_pairs_path = (
        output_directory / "near_duplicate_pairs.jsonl"
    )
    duplicate_pairs: list[dict[str, Any]] = []
    if bool(config.get("resume", True)) and duplicate_pairs_path.exists():
        duplicate_pairs.extend(read_jsonl(duplicate_pairs_path))
    file_reports: list[dict[str, Any]] = []

    for entry in priority:
        dataset_name = str(entry["dataset"])
        split = str(entry["split"])
        input_path = Path(entry["input_path"])

        if not input_path.exists():
            raise FileNotFoundError(
                f"Near-deduplication input does not exist: {input_path}"
            )

        dataset_output_directory = (
            output_directory / dataset_name
        )
        dataset_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            dataset_output_directory / f"{split}.jsonl"
        )

        report_path = output_path.with_suffix(
            ".near_deduplication.json"
        )

        # A report is written only after the output has been atomically
        # committed. Its presence therefore acts as a per-file checkpoint.
        if (
            bool(config.get("resume", True))
            and output_path.exists()
            and report_path.exists()
        ):
            with report_path.open("r", encoding="utf-8") as file:
                completed_statistics = json.load(file)
            file_reports.append(completed_statistics)
            LOGGER.info(
                "Checkpoint found; skipping completed %s/%s",
                dataset_name,
                split,
            )
            rebuild_statistics: dict[str, Any] = {
                "input_documents": 0,
                "output_documents": 0,
                "indexed_documents": 0,
                "evicted_index_documents": 0,
                "short_documents_not_indexed": 0,
                "candidate_matches": 0,
                "near_duplicate_documents": 0,
                "rejected_documents": 0,
                "rejection_counts": {},
            }
            # Rebuild only the bounded rolling state from checkpoint output.
            # This preserves cross-file comparisons without retaining the
            # complete corpus in memory or rerunning completed output writes.
            for _ in iter_near_unique_records(
                output_path,
                dataset_name=dataset_name,
                split=split,
                lsh=lsh,
                kept_shingles=kept_shingles,
                kept_records=kept_records,
                indexed_document_ids=indexed_document_ids,
                config=config,
                statistics=rebuild_statistics,
                duplicate_pairs=[],
            ):
                pass
            continue

        statistics: dict[str, Any] = {
            "dataset": dataset_name,
            "split": split,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "processed_at_utc": datetime.now(UTC).isoformat(),
            "input_documents": 0,
            "output_documents": 0,
            "indexed_documents": 0,
            "evicted_index_documents": 0,
            "short_documents_not_indexed": 0,
            "candidate_matches": 0,
            "near_duplicate_documents": 0,
            "rejected_documents": 0,
            "rejection_counts": {},
        }

        records = iter_near_unique_records(
            input_path,
            dataset_name=dataset_name,
            split=split,
            lsh=lsh,
            kept_shingles=kept_shingles,
            kept_records=kept_records,
            indexed_document_ids=indexed_document_ids,
            config=config,
            statistics=statistics,
            duplicate_pairs=duplicate_pairs,
        )

        temporary_output_path = output_path.with_suffix(
            output_path.suffix + ".partial"
        )
        written_count = write_jsonl(
            records,
            temporary_output_path,
        )

        if written_count != statistics["output_documents"]:
            raise RuntimeError(
                "Written-document count does not match "
                "near-deduplication statistics."
            )

        temporary_output_path.replace(output_path)

        temporary_report_path = report_path.with_suffix(
            report_path.suffix + ".partial"
        )
        with temporary_report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                statistics,
                file,
                indent=2,
                ensure_ascii=False,
            )
        temporary_report_path.replace(report_path)

        file_reports.append(statistics)

        LOGGER.info(
            "Near-deduplicated %s/%s: input=%d output=%d "
            "near_duplicates=%d",
            dataset_name,
            split,
            statistics["input_documents"],
            statistics["output_documents"],
            statistics["near_duplicate_documents"],
        )

    summary = {
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output_directory": str(output_directory),
        "configuration": {
            "shingle_size": int(config.get("shingle_size", 5)),
            "min_words": int(config.get("min_words", 20)),
            "num_permutations": num_permutations,
            "random_seed": int(config.get("random_seed", 42)),
            "lsh_threshold": lsh_threshold,
            "similarity_threshold": similarity_threshold,
            "max_indexed_documents": int(
                config.get("max_indexed_documents", 2000)
            ),
        },
        "input_documents": sum(
            report["input_documents"]
            for report in file_reports
        ),
        "output_documents": sum(
            report["output_documents"]
            for report in file_reports
        ),
        "indexed_documents": sum(
            report["indexed_documents"]
            for report in file_reports
        ),
        "evicted_index_documents": sum(
            report.get("evicted_index_documents", 0)
            for report in file_reports
        ),
        "short_documents_not_indexed": sum(
            report["short_documents_not_indexed"]
            for report in file_reports
        ),
        "candidate_matches": sum(
            report["candidate_matches"]
            for report in file_reports
        ),
        "near_duplicate_documents": sum(
            report["near_duplicate_documents"]
            for report in file_reports
        ),
        "rejected_documents": sum(
            report["rejected_documents"]
            for report in file_reports
        ),
        "files": file_reports,
    }

    summary_path = (
        output_directory
        / "near_deduplication_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if bool(config.get("write_duplicate_pairs", True)):
        write_jsonl(
            duplicate_pairs,
            duplicate_pairs_path,
        )

    LOGGER.info(
        "Global near-deduplication completed: "
        "input=%d output=%d removed=%d rejected=%d",
        summary["input_documents"],
        summary["output_documents"],
        summary["near_duplicate_documents"],
        summary["rejected_documents"],
    )

    return summary
