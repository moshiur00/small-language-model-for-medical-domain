"""Tests for bounded NCBI E-utilities ingestion."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from medical_slm.data.download import (
    _open_ncbi_request,
    iter_ncbi_xml_records,
    validate_download_limit_plan,
)


class FakeResponse:
    """Small context-manager response used to mock urlopen."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self.payload)

    def __exit__(self, *args: Any) -> None:
        return None


class InterruptedResponse:
    """Response whose body terminates before the final HTTP chunk."""

    def __enter__(self) -> "InterruptedResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        from http.client import IncompleteRead

        raise IncompleteRead(b"partial")


def test_ncbi_history_search_and_batched_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = {
        "esearchresult": {"count": "2", "querykey": "1", "webenv": "abc"}
    }
    xml = b"""
    <PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>2</PMID></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>
    """
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: int) -> FakeResponse:
        request_text = f"{request.full_url}?{request.data.decode()}"
        calls.append(request_text)
        payload = json.dumps(search).encode() if "esearch.fcgi" in request_text else xml
        return FakeResponse(payload)

    monkeypatch.setenv("NCBI_EMAIL", "researcher@example.org")
    monkeypatch.setattr("medical_slm.data.download.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("medical_slm.data.download.time.sleep", lambda _: None)

    records = list(
        iter_ncbi_xml_records(
            database="pubmed",
            search_term="hasabstract[text]",
            max_documents=2,
            batch_size=2,
            base_url="https://example.test",
            email_environment_variable="NCBI_EMAIL",
            api_key_environment_variable="NCBI_API_KEY",
        )
    )
    assert len(records) == 2
    assert "usehistory=y" in calls[0]
    assert "query_key=1" in calls[1]


def test_incomplete_ncbi_response_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[Any] = [InterruptedResponse(), FakeResponse(b"complete")]
    monkeypatch.setattr(
        "medical_slm.data.download.urllib.request.urlopen",
        lambda request, timeout: responses.pop(0),
    )
    monkeypatch.setattr("medical_slm.data.download.time.sleep", lambda _: None)

    payload = _open_ncbi_request(
        "https://example.test/efetch.fcgi",
        {"db": "pubmed"},
        timeout=1,
    )
    assert payload == b"complete"


def test_ncbi_requires_contact_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    with pytest.raises(ValueError, match="NCBI_EMAIL"):
        list(
            iter_ncbi_xml_records(
                database="pubmed",
                search_term="all[sb]",
                max_documents=1,
                batch_size=1,
                base_url="https://example.test",
                email_environment_variable="NCBI_EMAIL",
                api_key_environment_variable="NCBI_API_KEY",
            )
        )


def test_large_pubmed_download_is_partitioned_by_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terms: list[str] = []

    def fake_partition(**kwargs: Any):
        terms.append(kwargs["search_term"])
        yield {"xml": "<PubmedArticle />"}

    monkeypatch.setenv("NCBI_EMAIL", "researcher@example.org")
    monkeypatch.setattr(
        "medical_slm.data.download._iter_ncbi_history_records",
        fake_partition,
    )

    records = list(
        iter_ncbi_xml_records(
            database="pubmed",
            search_term="hasabstract[text]",
            max_documents=10_000,
            batch_size=200,
            base_url="https://example.test",
            email_environment_variable="NCBI_EMAIL",
            api_key_environment_variable="NCBI_API_KEY",
            date_start="2026/07/17",
            date_end="2026/07/18",
            date_field="CRDT",
        )
    )
    assert len(records) == 2
    assert '"2026/07/18"[CRDT]' in terms[0]
    assert '"2026/07/17"[CRDT]' in terms[1]


def test_download_limits_must_match_corpus_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "corpora.yaml"
    corpus_path.write_text(
        yaml.safe_dump(
            {"download_plan": {"example": {"max_documents": {"train": 10}}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        validate_download_limit_plan(
            root_config={"project": {"corpora_config": "corpora.yaml"}},
            dataset_name="example",
            dataset_config={"max_documents": {"train": 9}},
        )
