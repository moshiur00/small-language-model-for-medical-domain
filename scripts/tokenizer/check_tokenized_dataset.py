"""Check the generated packed-token dataset."""

from __future__ import annotations

from torch.utils.data import DataLoader

from medical_slm.data.tokenization import PackedTokenDataset


def main() -> None:
    train_dataset = PackedTokenDataset(
        "datasets/tokenized/train"
    )

    validation_dataset = PackedTokenDataset(
        "datasets/tokenized/validation"
    )

    test_dataset = PackedTokenDataset(
        "datasets/tokenized/test"
    )

    print("Train sequences:", len(train_dataset))
    print("Validation sequences:", len(validation_dataset))
    print("Test sequences:", len(test_dataset))

    sample = train_dataset[0]

    print("Input shape:", sample["input_ids"].shape)
    print("Label shape:", sample["labels"].shape)
    print("Attention-mask shape:", sample["attention_mask"].shape)

    print("First input IDs:", sample["input_ids"][:20])
    print("First label IDs:", sample["labels"][:20])

    loader = DataLoader(
        train_dataset,
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