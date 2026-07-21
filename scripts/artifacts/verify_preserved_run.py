"""Verify every file and checkpoint in a preserved training-run bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medical_slm.data.tokenization import calculate_sha256
from medical_slm.training.checkpoint import verify_checkpoint


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read preservation manifest: {path}") from error
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Unsupported preservation manifest.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    manifest = read_manifest(root / "preservation_manifest.json")

    for artifact in manifest["files"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe preserved artifact path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"Preserved artifact size mismatch: {relative}")
        if calculate_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Preserved artifact hash mismatch: {relative}")

    checkpoint_names = sorted(set(manifest["pointers"].values()))
    for checkpoint_name in checkpoint_names:
        verify_checkpoint(root / "checkpoints" / checkpoint_name)

    print(f"Preservation bundle verified: {root}")
    print(f"Files: {len(manifest['files'])}")
    print(f"Checkpoints: {', '.join(checkpoint_names)}")


if __name__ == "__main__":
    main()
