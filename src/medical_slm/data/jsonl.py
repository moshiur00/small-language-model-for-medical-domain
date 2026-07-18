"""Utilities for reading and writing JSON Lines files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Write records to a UTF-8 JSONL file.

    Args:
        records: Iterable of dictionaries.
        output_path: Destination JSONL file.

    Returns:
        Number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")
            count += 1

    return count


def read_jsonl(input_path: Path) -> Iterator[dict[str, Any]]:
    """
    Lazily read records from a JSONL file.

    Args:
        input_path: Source JSONL file.

    Yields:
        Parsed JSON objects.
    """
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {input_path}:{line_number}"
                ) from error