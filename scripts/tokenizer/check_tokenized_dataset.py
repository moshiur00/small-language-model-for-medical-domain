"""Check the generated packed-token dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from medical_slm.data.tokenization import PackedTokenDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/tokenized"),
    )
    parser.add_argument("--phase", action="append", default=None)
    arguments = parser.parse_args()

    phase_directories = sorted(
        path.parent
        for path in arguments.root.glob("*/dataset_manifest.json")
        if arguments.phase is None or path.parent.name in arguments.phase
    )
    if not phase_directories:
        raise FileNotFoundError("No phase tokenization manifests were found.")

    first_dataset = None
    for phase_directory in phase_directories:
        manifest = json.loads(
            (phase_directory / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        for split in manifest["splits"]:
            dataset = PackedTokenDataset(phase_directory / split)
            print(f"{phase_directory.name}/{split} sequences:", len(dataset))
            if first_dataset is None:
                first_dataset = dataset

    assert first_dataset is not None
    sample = first_dataset[0]

    print("Input shape:", sample["input_ids"].shape)
    print("Label shape:", sample["labels"].shape)
    print("Attention-mask shape:", sample["attention_mask"].shape)

    print("First input IDs:", sample["input_ids"][:20])
    print("First label IDs:", sample["labels"][:20])

    loader = DataLoader(
        first_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Batch input shape:", batch["input_ids"].shape)
    print("Batch label shape:", batch["labels"].shape)
    print(
        "Batch attention-mask shape:",
        batch["attention_mask"].shape,
    )


if __name__ == "__main__":
    main()
