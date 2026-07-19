"""NCBI PubMed XML standardizer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from medical_slm.data.standardizers.common import standardized_record


def _text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def standardize_pubmed_abstracts(
    dataset: Iterable[Mapping[str, Any]], **kwargs: Any
) -> Iterator[dict[str, Any]]:
    """Convert PubMedArticle XML fragments into title-plus-abstract documents."""
    written = 0
    limit = kwargs["max_documents"]
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        xml = example.get("xml")
        if not isinstance(xml, str):
            continue
        article = ET.fromstring(xml)
        pmid = _text(article.find(".//PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract_parts = []
        for part in article.findall(".//Abstract/AbstractText"):
            value = _text(part)
            label = part.attrib.get("Label")
            if value:
                abstract_parts.append(f"{label}: {value}" if label else value)
        abstract = "\n".join(abstract_parts)
        if not abstract:
            continue
        text = f"{title}\n\n{abstract}" if title else abstract
        yield standardized_record(
            source_index=index,
            text=text,
            document_type="biomedical_abstract",
            metadata={
                "pmid": pmid or None,
                "title": title or None,
                "journal": _text(article.find(".//Journal/Title")) or None,
                "doi": _doi(article),
                "copyright_information": (
                    _text(article.find(".//Abstract/CopyrightInformation")) or None
                ),
            },
            **{key: kwargs[key] for key in _STANDARD_KEYS},
        )
        written += 1


def _doi(article: ET.Element) -> str | None:
    for identifier in article.findall(".//ArticleId"):
        if identifier.attrib.get("IdType") == "doi":
            return _text(identifier) or None
    return None


_STANDARD_KEYS = (
    "hub_name", "config_name", "source", "source_split", "output_split",
    "license_name", "language",
)
