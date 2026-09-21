import torch

from src.dataset import SFTDataset

SAMPLES = [
    {
        "instruction": "Say hello.",
        "input": "",
        "output": "Hello!",
    },
    {
        "instruction": "Translate the text.",
        "input": "Hello",
        "output": "你好",
    },
]

class FakeTokenizer(object):
    eos_token = "<eos>"
    eos_token_id = 999_999

    def __init__(self):
        self.last_text = None

    def __call__(
            self,
            text,
            truncation,
            max_length,
            padding,
            return_tensors,
    ):
        self.last_text = text

        assert padding is False
        assert return_tensors == "pt"

        has_eos = text.endswith(self.eos_token)
        content = text.removesuffix(self.eos_token)

        token_ids = list(range(1, len(content) + 1))

        if has_eos:
            token_ids.append(self.eos_token_id)

        if truncation:
            token_ids = token_ids[:max_length]

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        ).unsqueeze(0)

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

def build_dataset(max_length=512):
    tokenizer = FakeTokenizer()

    dataset = SFTDataset(
        dataset=SAMPLES,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return dataset, tokenizer

def test_dataset_length():
    dataset, _ = build_dataset()

    assert len(dataset) == 2


def test_empty_input_format():
    dataset, tokenizer = build_dataset()

    dataset[0]

    expected_text = (
        "### Instruction:\n"
        "Say hello.\n\n"
        "### Response:\n"
        "Hello!\n\n"
        "<eos>"
    )

    assert tokenizer.last_text == expected_text
    assert "### Input:" not in tokenizer.last_text

def test_nonempty_input_format():
    dataset, tokenizer = build_dataset()

    dataset[1]

    expected_text = (
        "### Instruction:\n"
        "Translate the text.\n\n"
        "### Input:\n"
        "Hello\n\n"
        "### Response:\n"
        "你好\n\n"
        "<eos>"
    )

    assert tokenizer.last_text == expected_text
    assert "### Input:" in tokenizer.last_text

def test_item_tensor_contract():
    dataset, _ = build_dataset()
    item = dataset[0]

    assert set(item.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
    }

    assert item["input_ids"].ndim == 1
    assert item["attention_mask"].ndim == 1
    assert item["labels"].ndim == 1

    assert item["input_ids"].shape == item["attention_mask"].shape
    assert item["input_ids"].shape == item["labels"].shape

    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.long

    assert torch.equal(item["input_ids"], item["labels"])
    assert item["input_ids"].data_ptr() != item["labels"].data_ptr()

def test_eos_and_no_fixed_padding():
    dataset, tokenizer = build_dataset(max_length=512)
    item = dataset[0]

    assert item["input_ids"][-1].item() == tokenizer.eos_token_id
    assert item["attention_mask"][-1].item() == 1
    assert len(item["input_ids"]) < 512

def test_truncation():
    dataset, _ = build_dataset(max_length=16)
    item = dataset[0]

    assert len(item["input_ids"]) == 16
    assert len(item["attention_mask"]) == 16
    assert len(item["labels"]) == 16