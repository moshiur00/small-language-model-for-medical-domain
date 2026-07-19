"""Schema tests for every newly supported source-specific standardizer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import pytest

from medical_slm.data.standardizers import (
    standardize_alpaca,
    standardize_chatdoctor,
    standardize_fineweb_edu,
    standardize_medalpaca,
    standardize_medinstruct,
    standardize_medmcqa,
    standardize_openmedinstruct,
    standardize_pmc_open_access,
    standardize_project_gutenberg_public_domain,
    standardize_pubmed_abstracts,
    standardize_pubmedqa,
    standardize_wikidoc,
)


COMMON = {
    "hub_name": "organization/dataset",
    "config_name": None,
    "source": "example",
    "source_split": "train",
    "output_split": "train",
    "license_name": "test-license",
    "language": "en",
    "max_documents": None,
}

Standardizer = Callable[..., Iterator[dict[str, Any]]]


@pytest.mark.parametrize(
    ("standardizer", "example", "expected"),
    [
        (
            standardize_fineweb_edu,
            {"text": "Educational page", "token_count": 10, "score": 4.0},
            "Educational page",
        ),
        (
            standardize_project_gutenberg_public_domain,
            {"text": "Public-domain book", "metadata": {"book_id": "1"}},
            "Public-domain book",
        ),
        (
            standardize_wikidoc,
            {"source": "wikidoc", "title": "Heart", "clean_text": "Clinical text"},
            "Heart\n\nClinical text",
        ),
        (
            standardize_alpaca,
            {"instruction": "Do this", "input": "Context", "output": "Done"},
            "Response:\nDone",
        ),
        (
            standardize_medalpaca,
            {"instruction": "Answer truthfully", "input": "Question", "output": "Answer"},
            "Response:\nAnswer",
        ),
        (
            standardize_chatdoctor,
            {"instruction": "Patient", "input": "Symptoms", "output": "Advice"},
            "Response:\nAdvice",
        ),
        (
            standardize_medinstruct,
            {"instruction": "Medical task", "text-davinci-003-answer": "Response"},
            "Response:\nResponse",
        ),
        (
            standardize_openmedinstruct,
            {
                "messages": [
                    {"role": "user", "content": "Medical query"},
                    {"role": "assistant", "content": "Medical response"},
                ]
            },
            "Response:\nMedical response",
        ),
        (
            standardize_pubmedqa,
            {
                "pubid": 1,
                "question": "Is it effective?",
                "context": {"contexts": ["Study context."]},
                "long_answer": "It was effective.",
                "final_decision": "yes",
            },
            "It was effective.",
        ),
    ],
)
def test_source_standardizer_schema(
    standardizer: Standardizer,
    example: Mapping[str, Any],
    expected: str,
) -> None:
    records = list(standardizer([example], **COMMON))
    assert len(records) == 1
    assert expected in records[0]["text"]
    assert records[0]["source"] == "example"
    assert isinstance(records[0]["metadata"], dict)


def test_medmcqa_standardizer() -> None:
    record = list(
        standardize_medmcqa(
            [{
                "question": "Which?", "opa": "A1", "opb": "B1",
                "opc": "C1", "opd": "D1", "cop": 2, "exp": "Reason",
            }],
            **COMMON,
        )
    )[0]
    assert "C. C1" in record["text"]
    assert record["metadata"]["answer_index"] == 2


def test_pubmed_abstract_standardizer() -> None:
    xml = """
    <PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
      <ArticleTitle>Trial</ArticleTitle><Abstract>
        <AbstractText Label="RESULTS">Useful result.</AbstractText>
      </Abstract><Journal><Title>Journal</Title></Journal>
    </Article></MedlineCitation></PubmedArticle>
    """
    record = list(standardize_pubmed_abstracts([{"xml": xml}], **COMMON))[0]
    assert record["metadata"]["pmid"] == "123"
    assert "RESULTS: Useful result." in record["text"]


def test_pmc_standardizer_uses_article_license() -> None:
    xml = """
    <article><front><article-meta>
      <article-id pub-id-type="pmc">PMC1</article-id>
      <title-group><article-title>Open article</article-title></title-group>
      <permissions><license>CC BY 4.0 https://creativecommons.org/licenses/by/4.0/</license></permissions>
    </article-meta></front><body><sec><p>Medical full text.</p></sec></body></article>
    """
    record = list(standardize_pmc_open_access([{"xml": xml}], **COMMON))[0]
    assert record["license"] == "cc-by-4.0"
    assert record["metadata"]["pmcid"] == "PMC1"


def test_standardizers_obey_output_limit() -> None:
    examples: Iterable[Mapping[str, Any]] = ({"text": f"Document {i}"} for i in range(5))
    records = list(standardize_fineweb_edu(examples, **{**COMMON, "max_documents": 2}))
    assert len(records) == 2
