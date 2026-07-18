"""WikiText-103 dataset standardizer."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from tqdm import tqdm

from medical_slm.data.download import create_document_id


LOGGER = logging.getLogger(__name__)

# Supports both conventional headings:
#
#   = Article title =
#   == Section title ==
#   === Subsection title ===
#
# and WikiText headings containing spaces between equal signs:
#
#   = = Section title = =
#   = = = Subsection title = = =
#
HEADING_PATTERN = re.compile(
    r"^\s*"
    r"(?P<left>(?:=\s*)+)"
    r"(?P<title>.*?)"
    r"(?P<right>(?:\s*=)+)"
    r"\s*$"
)


def extract_heading(text: str) -> tuple[str, int] | None:
    """
    Extract a WikiText heading and its level.

    Examples:
        ``= Article title =`` returns ``("Article title", 1)``.
        ``== Section ==`` returns ``("Section", 2)``.
        ``= = Section = =`` returns ``("Section", 2)``.

    Args:
        text:
            Source WikiText row.

    Returns:
        A ``(heading, level)`` tuple, or ``None`` when the row is not
        a valid heading.
    """
    if not isinstance(text, str):
        return None

    match = HEADING_PATTERN.fullmatch(text)

    if match is None:
        return None

    left_marker = match.group("left")
    right_marker = match.group("right")
    title = match.group("title").strip()

    left_level = left_marker.count("=")
    right_level = right_marker.count("=")

    if not title:
        return None

    if left_level != right_level:
        return None

    return title, left_level


def format_heading(
    heading: str,
    *,
    level: int,
) -> str:
    """
    Format a heading for model-visible article text.

    Equal-sign markup is removed because the heading text itself carries
    the useful semantic information.
    """
    del level
    return heading.strip()


def build_article_text(
    *,
    title: str | None,
    content_parts: Sequence[str],
) -> str:
    """
    Build one article-level document from a title and its content rows.

    Duplicate adjacent parts are not introduced, and blank content parts
    are ignored.
    """
    normalized_parts: list[str] = []

    if title:
        normalized_title = title.strip()

        if normalized_title:
            normalized_parts.append(normalized_title)

    for part in content_parts:
        normalized_part = part.strip()

        if not normalized_part:
            continue

        if (
            normalized_parts
            and normalized_parts[-1] == normalized_part
        ):
            continue

        normalized_parts.append(normalized_part)

    return "\n\n".join(normalized_parts).strip()


def create_article_record(
    *,
    title: str | None,
    content_parts: Sequence[str],
    section_headings: Sequence[dict[str, Any]],
    hub_name: str,
    config_name: str | None,
    source: str,
    source_split: str,
    output_split: str,
    license_name: str,
    language: str,
    source_start_index: int,
    source_end_index: int,
) -> dict[str, Any] | None:
    """
    Create one standardized article record.

    Returns ``None`` when the buffered article has no usable text.
    """
    article_text = build_article_text(
        title=title,
        content_parts=content_parts,
    )

    if not article_text:
        return None

    return {
        "id": create_document_id(
            source,
            output_split,
            source_start_index,
            article_text,
        ),
        "source": source,
        "source_dataset": hub_name,
        "source_config": config_name,
        "source_split": source_split,
        "license": license_name,
        "language": language,
        "text": article_text,
        "metadata": {
            "source_start_index": source_start_index,
            "source_end_index": source_end_index,
            "source_row_count": (
                source_end_index - source_start_index + 1
            ),
            "document_type": "wikitext_article",
            "title": title,
            "section_count": len(section_headings),
            "section_headings": list(section_headings),
        },
    }


def standardize_wikitext(
    dataset: Iterable[Mapping[str, Any]],
    *,
    hub_name: str,
    config_name: str | None,
    source: str,
    source_split: str,
    output_split: str,
    license_name: str,
    language: str,
    max_documents: int | None,
) -> Iterator[dict[str, Any]]:
    """
    Reconstruct WikiText-103 rows into article-level documents.

    A level-one heading starts a new article. Lower-level headings and
    paragraphs are appended to the current article. The completed article
    is emitted when the next level-one heading is encountered.

    ``max_documents`` limits the number of reconstructed articles written,
    not the number of source rows consumed.
    """
    progress = tqdm(
        dataset,
        desc=f"Reconstructing {source}/{output_split}",
        unit="rows",
    )

    current_title: str | None = None
    current_content_parts: list[str] = []
    current_section_headings: list[dict[str, Any]] = []
    current_start_index: int | None = None
    current_end_index: int | None = None

    written_count = 0

    for source_index, example in enumerate(progress):
        text = example.get("text")

        if not isinstance(text, str):
            LOGGER.warning(
                "Skipping WikiText row %d: text is not a string.",
                source_index,
            )
            continue

        text = text.strip()

        if not text:
            continue

        heading = extract_heading(text)

        if heading is not None:
            heading_text, heading_level = heading

            # A level-one heading marks the beginning of a new article.
            if heading_level == 1:
                if current_start_index is not None:
                    record = create_article_record(
                        title=current_title,
                        content_parts=current_content_parts,
                        section_headings=current_section_headings,
                        hub_name=hub_name,
                        config_name=config_name,
                        source=source,
                        source_split=source_split,
                        output_split=output_split,
                        license_name=license_name,
                        language=language,
                        source_start_index=current_start_index,
                        source_end_index=(
                            current_end_index
                            if current_end_index is not None
                            else current_start_index
                        ),
                    )

                    if record is not None:
                        yield record
                        written_count += 1

                        progress.set_postfix(
                            articles=written_count,
                            refresh=False,
                        )

                        if (
                            max_documents is not None
                            and written_count >= max_documents
                        ):
                            break

                current_title = heading_text
                current_content_parts = []
                current_section_headings = []
                current_start_index = source_index
                current_end_index = source_index
                continue

            # Lower-level headings belong to the current article.
            if current_start_index is None:
                current_start_index = source_index

            formatted_heading = format_heading(
                heading_text,
                level=heading_level,
            )

            current_content_parts.append(formatted_heading)
            current_section_headings.append(
                {
                    "heading": heading_text,
                    "level": heading_level,
                    "source_index": source_index,
                }
            )
            current_end_index = source_index
            continue

        # A normal text row belongs to the current article. If WikiText
        # contains text before its first top-level heading, retain it as an
        # untitled document rather than silently discarding it.
        if current_start_index is None:
            current_start_index = source_index

        current_content_parts.append(text)
        current_end_index = source_index

    else:
        # The loop ended naturally, so emit the final buffered article.
        if current_start_index is not None:
            record = create_article_record(
                title=current_title,
                content_parts=current_content_parts,
                section_headings=current_section_headings,
                hub_name=hub_name,
                config_name=config_name,
                source=source,
                source_split=source_split,
                output_split=output_split,
                license_name=license_name,
                language=language,
                source_start_index=current_start_index,
                source_end_index=(
                    current_end_index
                    if current_end_index is not None
                    else current_start_index
                ),
            )

            if (
                record is not None
                and (
                    max_documents is None
                    or written_count < max_documents
                )
            ):
                yield record