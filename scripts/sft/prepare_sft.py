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
    sft_config = config.get("sft_preparation")
    if not isinstance(sft_config, dict):
        raise ValueError("Configuration must contain 'sft_preparation'.")
    manifest = prepare_sft_dataset(sft_config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
