"""Prepare structured, response-masked SFT training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from medical_slm.data.sft.pipeline import prepare_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SFT configuration root must be a mapping.")
    nested = config.get("sft_preparation")
    sft_config = nested if nested is not None else config
    if not isinstance(sft_config, dict):
        raise ValueError("'sft_preparation' must be a mapping.")
    manifest = prepare_sft_dataset(sft_config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
