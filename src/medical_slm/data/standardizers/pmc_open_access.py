"""PMC Open Access NXML standardizer with article-level licensing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import standardized_record


LICENSE_PATTERNS = (
    (re.compile(r"creativecommons\.org/publicdomain/zero|\bcc0\b", re.I), "cc0-1.0"),
    (re.compile(r"creativecommons\.org/licenses/by/4|\bcc by 4", re.I), "cc-by-4.0"),
    (re.compile(r"creativecommons\.org/licenses/by/3|\bcc by 3", re.I), "cc-by-3.0"),
    (re.compile(r"creativecommons\.org/licenses/by-sa", re.I), "cc-by-sa"),
    (re.compile(r"creativecommons\.org/licenses/by-nc", re.I), "cc-by-nc"),
)


def _text(element: ET.Element | None) -> str:
    return "" if element is None else " ".join("".join(element.itertext()).split())


def _article_license(article: ET.Element) -> tuple[str, str]:
    statement = _text(article.find(".//permissions/license"))
    for pattern, identifier in LICENSE_PATTERNS:
        if pattern.search(statement):
            return identifier, statement
    return "unknown", statement


def standardize_pmc_open_access(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert PMC articles and override the blanket license per article."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        xml = example.get("xml")
        if not isinstance(xml, str):
            continue
        article = ET.fromstring(xml)
        title = _text(article.find(".//article-title"))
        paragraphs = [_text(item) for item in article.findall(".//body//p")]
        body = "\n\n".join(value for value in paragraphs if value)
        if not body:
            continue
        license_name, license_statement = _article_license(article)
        text = f"{title}\n\n{body}" if title else body
        record = standardized_record(
            source_index=index,
            text=text,
            document_type="biomedical_full_text_article",
            metadata={
                "pmcid": _text(article.find(".//article-id[@pub-id-type='pmc']")) or None,
                "pmid": _text(article.find(".//article-id[@pub-id-type='pmid']")) or None,
                "doi": _text(article.find(".//article-id[@pub-id-type='doi']")) or None,
                "title": title or None,
                "license_statement": license_statement or None,
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        record["license"] = license_name
        yield record
        written += 1


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)
