import torch

from src.collator import SFTDataCollator


def make_variable_length_features():
    return [
        {
            "input_ids": torch.tensor([11, 12], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1], dtype=torch.long),
            "labels": torch.tensor([11, 12], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([21, 22, 23, 24], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([21, 22, 23, 24], dtype=torch.long),
        },
    ]


def test_collator_pads_variable_length_features():
    collator = SFTDataCollator(pad_token_id=99)

    batch = collator(make_variable_length_features())

    expected_input_ids = torch.tensor(
        [
            [11, 12, 99, 99],
            [21, 22, 23, 24],
        ],
        dtype=torch.long,
    )
    expected_attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    expected_labels = torch.tensor(
        [
            [11, 12, -100, -100],
            [21, 22, 23, 24],
        ],
        dtype=torch.long,
    )

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == torch.Size([2, 4])
    assert batch["attention_mask"].shape == torch.Size([2, 4])
    assert batch["labels"].shape == torch.Size([2, 4])
    assert torch.equal(batch["input_ids"], expected_input_ids)
    assert torch.equal(batch["attention_mask"], expected_attention_mask)
    assert torch.equal(batch["labels"], expected_labels)


def test_collator_does_not_modify_features():
    features = make_variable_length_features()
    originals = [
        {key: tensor.clone() for key, tensor in feature.items()}
        for feature in features
    ]
    collator = SFTDataCollator(pad_token_id=99)

    collator(features)

    for feature, original in zip(features, originals):
        for key in feature:
            assert torch.equal(feature[key], original[key])


def test_collator_does_not_pad_equal_length_features():
    features = [
        {
            "input_ids": torch.tensor([11, 12]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([11, 12]),
        },
        {
            "input_ids": torch.tensor([21, 22]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([21, 22]),
        },
    ]
    collator = SFTDataCollator(pad_token_id=99)

    batch = collator(features)

    assert batch["input_ids"].shape == torch.Size([2, 2])
    assert torch.equal(
        batch["input_ids"],
        torch.tensor([[11, 12], [21, 22]]),
    )
    assert torch.all(batch["attention_mask"] == 1)
    assert torch.all(batch["labels"] != -100)


def test_collator_handles_single_feature_batch():
    features = [
        {
            "input_ids": torch.tensor([11, 12, 13]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([11, 12, 13]),
        }
    ]
    collator = SFTDataCollator(pad_token_id=99)

    batch = collator(features)

    assert batch["input_ids"].shape == torch.Size([1, 3])
    assert batch["attention_mask"].shape == torch.Size([1, 3])
    assert batch["labels"].shape == torch.Size([1, 3])
    assert torch.equal(batch["input_ids"][0], features[0]["input_ids"])
    assert torch.equal(
        batch["attention_mask"][0],
        features[0]["attention_mask"],
    )
    assert torch.equal(batch["labels"][0], features[0]["labels"])
