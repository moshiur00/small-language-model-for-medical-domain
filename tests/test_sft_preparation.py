"""Tests for structured response-only SFT preparation."""

from medical_slm.data.sft.pipeline import (
    IGNORE_INDEX,
    create_response_masked_example,
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
