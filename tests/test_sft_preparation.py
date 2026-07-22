"""Tests for structured response-only SFT preparation."""

import pytest

from medical_slm.data.sft.pipeline import (
    IGNORE_INDEX,
    _deduplicate_records,
    _split_records,
    create_response_masked_example,
    create_structured_response_masked_example,
    parse_sft_text,
)


class FakeTokenizer:
    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(range(10, 10 + len(text.split())))


def test_parse_instruction_input_response() -> None:
    parsed = parse_sft_text(
        "Instruction:\nQuestion?\n\nInput:\nContext\n\nResponse:\nAnswer"
    )
    assert parsed["instruction"] == "Question?"
    assert parsed["context"] == "Context"
    assert parsed["response"] == "Answer"


def test_parse_tolerates_cleaning_variants() -> None:
    parsed = parse_sft_text(
        "Instruction:. Question?\n\nOptions:\nA. One\nB. Two"
        "\n\nResponse:B. Two"
    )
    assert parsed["instruction"] == "Question?"
    assert parsed["context_type"] == "options"
    assert parsed["response"] == "B. Two"


def test_only_response_and_eos_are_supervised() -> None:
    input_ids, attention_mask, labels, truncated = create_response_masked_example(
        FakeTokenizer(), prompt="Instruction question Response", response="answer now",
        max_length=10,
    )
    assert not truncated
    response_start = 4
    assert labels[:response_start] == [IGNORE_INDEX] * response_start
    assert labels[response_start:7] == input_ids[response_start:7]
    assert labels[7:] == [IGNORE_INDEX] * 3
    assert attention_mask == [1] * 7 + [0] * 3


def test_long_response_is_truncated_without_removing_prompt() -> None:
    input_ids, attention_mask, labels, truncated = create_response_masked_example(
        FakeTokenizer(),
        prompt="instruction must remain",
        response="one two three four five six seven eight",
        max_length=8,
    )
    assert truncated
    assert labels[:4] == [IGNORE_INDEX] * 4
    assert labels[4:8] == input_ids[4:8]
    assert attention_mask == [1] * 8


def test_structured_truncation_removes_context_before_response() -> None:
    encoded = create_structured_response_masked_example(
        FakeTokenizer(),
        instruction="answer this question",
        context="long context that cannot all fit here",
        context_type="input",
        response="answer remains",
        max_length=10,
    )
    assert encoded.context_truncated
    assert not encoded.response_truncated
    assert encoded.response_tokens == 3  # Two response tokens plus EOS.
    assert encoded.labels[: 1 + encoded.prompt_tokens] == [IGNORE_INDEX] * (
        1 + encoded.prompt_tokens
    )


def test_structured_truncation_rejects_instruction_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="instruction and response separator"):
        create_structured_response_masked_example(
            FakeTokenizer(),
            instruction="one two three four five six seven",
            context="",
            context_type="none",
            response="answer",
            max_length=6,
        )


def test_normalized_duplicate_pairs_and_ids_are_removed() -> None:
    records = [
        {
            "id": "one",
            "source": "a",
            "instruction": "What is BP?",
            "context": "",
            "context_type": "none",
            "response": "Blood pressure",
        },
        {
            "id": "two",
            "source": "b",
            "instruction": "  WHAT   is bp? ",
            "context": "",
            "context_type": "none",
            "response": " blood PRESSURE ",
        },
        {
            "id": "one",
            "source": "c",
            "instruction": "Different prompt",
            "context": "",
            "context_type": "none",
            "response": "Different response",
        },
    ]
    accepted, rejected = _deduplicate_records(records)
    assert [record["id"] for record in accepted] == ["one"]
    assert {record["rejection_reason"] for record in rejected} == {
        "duplicate normalized prompt-response pair",
        "duplicate record ID",
    }


def test_prompt_groups_never_cross_deterministic_splits() -> None:
    records = []
    for index in range(200):
        group = f"group-{index // 2}"
        records.append(
            {
                "id": f"id-{index}",
                "source": f"source-{index % 7}",
                "split_group_sha256": group,
            }
        )
    first = _split_records(records, 0.05, 0.05, 42)
    second = _split_records(records, 0.05, 0.05, 42)
    assert {
        split: [record["id"] for record in values]
        for split, values in first.items()
    } == {
        split: [record["id"] for record in values]
        for split, values in second.items()
    }
    assert all(first[split] for split in ("train", "validation", "test"))
    groups = {
        split: {record["split_group_sha256"] for record in values}
        for split, values in first.items()
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
