"""Tokenizer training, evaluation and comparison utilities."""

from medical_slm.tokenizer.compare import (
    DEFAULT_GPT2_TOKENIZER_NAME,
    DEFAULT_MEDICAL_TERMS,
    LengthStatistics,
    MedicalTermResult,
    TokenizerComparisonResult,
    TokenizerEvaluationResult,
    build_difference_report,
    build_recommendation,
    compare_tokenizers,
    encode_complete_text,
    evaluate_medical_terms,
    evaluate_single_tokenizer,
    iter_jsonl_texts,
    write_json_report,
    write_markdown_report,
)
from medical_slm.tokenizer.evaluate import (
    TokenizerMetricsAccumulator,
    encode_without_special_tokens,
    evaluate_tokenizer,
)
from medical_slm.tokenizer.manifest import (
    calculate_file_sha256,
    write_tokenizer_manifest,
)
from medical_slm.tokenizer.train import (
    collect_tokenizer_artifacts,
    configure_bos_eos_processor,
    create_byte_level_bpe_tokenizer,
    get_special_tokens,
    train_byte_level_bpe,
    validate_saved_tokenizer,
    validate_tokenizer_config,
)


__all__ = [
    "DEFAULT_GPT2_TOKENIZER_NAME",
    "DEFAULT_MEDICAL_TERMS",
    "LengthStatistics",
    "MedicalTermResult",
    "TokenizerComparisonResult",
    "TokenizerEvaluationResult",
    "TokenizerMetricsAccumulator",
    "build_difference_report",
    "build_recommendation",
    "calculate_file_sha256",
    "collect_tokenizer_artifacts",
    "compare_tokenizers",
    "configure_bos_eos_processor",
    "create_byte_level_bpe_tokenizer",
    "encode_complete_text",
    "encode_without_special_tokens",
    "evaluate_medical_terms",
    "evaluate_single_tokenizer",
    "evaluate_tokenizer",
    "get_special_tokens",
    "iter_jsonl_texts",
    "train_byte_level_bpe",
    "validate_saved_tokenizer",
    "validate_tokenizer_config",
    "write_json_report",
    "write_markdown_report",
    "write_tokenizer_manifest",
]