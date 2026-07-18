"""Dataset deduplication utilities."""

from medical_slm.data.deduplication.exact import (
    add_deduplication_metadata,
    canonicalize_text,
    create_content_hash,
    deduplicate_dataset_splits,
    deduplicate_jsonl_file,
    get_record_content_hash,
)
from medical_slm.data.deduplication.global_exact import (
    run_global_exact_deduplication,
)
from medical_slm.data.deduplication.minhash import (
    calculate_jaccard_similarity,
    canonicalize_for_near_deduplication,
    create_minhash,
    create_word_shingles,
    run_global_near_deduplication,
    tokenize_words,
)

__all__ = [
    "add_deduplication_metadata",
    "calculate_jaccard_similarity",
    "canonicalize_for_near_deduplication",
    "canonicalize_text",
    "create_content_hash",
    "create_minhash",
    "create_word_shingles",
    "deduplicate_dataset_splits",
    "deduplicate_jsonl_file",
    "get_record_content_hash",
    "run_global_exact_deduplication",
    "run_global_near_deduplication",
    "tokenize_words",
]