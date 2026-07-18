"""Language-identification and verification utilities."""

from medical_slm.data.language.detector import (
    FastTextLanguageDetector,
    LanguageDetector,
    LanguagePrediction,
    normalize_fasttext_label,
    prepare_text_for_prediction,
)
from medical_slm.data.language.pipeline import (
    classify_language_result,
    run_language_verification,
    validate_language_config,
    verify_jsonl_language,
)

__all__ = [
    "FastTextLanguageDetector",
    "LanguageDetector",
    "LanguagePrediction",
    "classify_language_result",
    "normalize_fasttext_label",
    "prepare_text_for_prediction",
    "run_language_verification",
    "validate_language_config",
    "verify_jsonl_language",
]