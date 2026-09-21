from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "alpaca_seed42_train2000_val200"
)


def build_data_splits(
        raw_dataset,
        train_size=2000,
        val_size=200,
        seed=42
):
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")

    if train_size + val_size > len(raw_dataset):
        raise ValueError("not enough samples for the requested split")

    indexed_dataset = raw_dataset.add_column(
        "sample_id",
        list(range(len(raw_dataset))),
    )

    shuffled_dataset = indexed_dataset.shuffle(seed=seed)

    val_dataset = shuffled_dataset.select(
        range(val_size)
    )

    train_dataset = shuffled_dataset.select(
        range(val_size, val_size + train_size)
    )

    return train_dataset, val_dataset

def main():
    raw_dataset = load_dataset(
        "yahma/alpaca-cleaned",
        split="train",
    )

    train_dataset, val_dataset = build_data_splits(
        raw_dataset,
        train_size=2000,
        val_size=200,
        seed=42
    )

    train_again, val_again = build_data_splits(
        raw_dataset,
        train_size=2000,
        val_size=200,
        seed=42,
    )

    train_ids = list(train_dataset["sample_id"])
    val_ids = list(val_dataset["sample_id"])

    assert len(train_dataset) == 2000
    assert len(val_dataset) == 200

    assert set(train_ids).isdisjoint(val_ids)

    assert train_ids == list(train_again["sample_id"])
    assert val_ids == list(val_again["sample_id"])

    assert "sample_id" not in raw_dataset.column_names

    print("train samples:", len(train_dataset))
    print("validation samples:", len(val_dataset))
    print("first 5 train IDs:", train_ids[:5])
    print("first 5 validation IDs:", val_ids[:5])
    print("Data split verification passed.")

    splits = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    if DATA_DIR.exists():
        print("Checking existing saved data:", DATA_DIR)
    else:
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(DATA_DIR))
        print("Data saved:", DATA_DIR)

    restored = load_from_disk(str(DATA_DIR))

    assert set(restored.keys()) == {"train", "validation"}

    for split_name in ("train", "validation"):
        original = splits[split_name]
        loaded = restored[split_name]

        assert len(loaded) == len(original)
        assert loaded.features == original.features
        assert loaded[:] == original[:]

        print(f"{split_name}: {len(loaded)} samples verified")

    print("Saved data verification passed.")

if __name__ == "__main__":
    main()

