from pathlib import Path

from medical_slm.data.jsonl import read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        {"id": "one", "text": "First document."},
        {"id": "two", "text": "Second document."},
    ]

    output_path = tmp_path / "records.jsonl"

    count = write_jsonl(records, output_path)
    loaded_records = list(read_jsonl(output_path))

    assert count == 2
    assert loaded_records == records