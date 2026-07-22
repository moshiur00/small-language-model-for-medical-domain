"""Tests for Stage C near-duplicate and license auditing."""

from scripts.sft.audit_stage_c_dataset import (
    find_near_duplicate_pairs,
    summarize_licenses,
)


def record(
    identifier: str,
    *,
    split: str,
    source: str,
    instruction: str,
    context: str = "",
    decision: str = "pass",
    commercial_use: bool = True,
    redistribution: bool = True,
) -> dict[str, object]:
    return {
        "id": identifier,
        "split": split,
        "source": source,
        "instruction": instruction,
        "context": context,
        "context_type": "input" if context else "none",
        "split_group_sha256": f"group-{identifier}",
        "license": "test-license",
        "metadata": {
            "license_validation": {
                "decision": decision,
                "obligations": {
                    "commercial_use": commercial_use,
                    "redistribution": redistribution,
                },
            }
        },
    }


def test_near_duplicate_audit_detects_cross_split_and_source_pair() -> None:
    first = record(
        "first",
        split="train",
        source="source-a",
        instruction="Explain the treatment of severe bacterial pneumonia in adults",
        context="Use current clinical evidence and discuss antibiotic therapy.",
    )
    second = record(
        "second",
        split="validation",
        source="source-b",
        instruction="Explain treatment of severe bacterial pneumonia in adults",
        context="Use current clinical evidence and discuss antibiotic therapy.",
    )
    result = find_near_duplicate_pairs(
        [first, second],
        shingle_size=2,
        lsh_threshold=0.5,
        similarity_threshold=0.75,
    )
    assert result["near_duplicate_pairs"] == 1
    assert result["cross_source_pairs"] == 1
    assert result["cross_split_pairs"] == 1
    assert result["cross_split_clusters"] == 1


def test_exact_prompt_group_is_not_reported_as_new_near_duplicate() -> None:
    first = record(
        "first",
        split="train",
        source="source-a",
        instruction="What is hypertension?",
    )
    second = record(
        "second",
        split="train",
        source="source-a",
        instruction="What is hypertension?",
    )
    second["split_group_sha256"] = first["split_group_sha256"]
    result = find_near_duplicate_pairs(
        [first, second],
        shingle_size=2,
        lsh_threshold=0.5,
        similarity_threshold=0.75,
    )
    assert result["near_duplicate_pairs"] == 0


def test_license_audit_allows_research_but_blocks_public_release() -> None:
    records = [
        record(
            "pass",
            split="train",
            source="permissive",
            instruction="Question one",
        ),
        record(
            "review",
            split="train",
            source="noncommercial",
            instruction="Question two",
            decision="review",
            commercial_use=False,
        ),
    ]
    result = summarize_licenses(records)
    assert result["internal_research_training_status"] == "passed"
    assert result["manual_review_examples"] == 1
    assert result["public_model_release_status"] == "blocked_pending_manual_review"


def test_license_audit_blocks_invalid_decision() -> None:
    result = summarize_licenses(
        [
            record(
                "blocked",
                split="train",
                source="unknown",
                instruction="Question",
                decision="fail",
            )
        ]
    )
    assert result["internal_research_training_status"] == "blocked"
