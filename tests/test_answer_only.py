"""Offline offset-boundary tests plus Dataset/Collator/loss integration."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.collator import SFTDataCollator
from src.dataset import SFTDataset
from src.evaluation import evaluation_loss


PROMPT = "### Instruction:\nReply.\n\n### Response:\n"
SAMPLE = {"instruction": " Reply. ", "input": "", "output": " ANSWER "}


class OffsetTokenizer:
    """One character per token; optional merged token straddles the boundary.

    BOS and the extra trailing special token have zero-width offsets. Literal
    EOS spans its text, as in the tokenizer used for this project's Qwen runs.
    """

    eos_token = "<eos>"
    eos_token_id = 9
    pad_token_id = 0

    def __init__(self, merge_boundary=False, eos=True):
        self.merge_boundary = merge_boundary
        if not eos:
            self.eos_token = None

    def __call__(self, text, truncation, max_length, padding, return_tensors,
                 return_offsets_mapping=False):
        assert truncation and not padding and return_tensors == "pt"
        content = text.removesuffix(self.eos_token) if self.eos_token else text
        records = [(0, (0, 0))]
        merge_start = text.index("ANSWER") - 1 if self.merge_boundary else -1
        index = 0
        while index < len(content):
            if index == merge_start:
                records.append((11, (index, index + 2)))
                index += 2
            else:
                records.append((ord(content[index]) % 7 + 1, (index, index + 1)))
                index += 1
        if self.eos_token:
            records.append((self.eos_token_id, (len(content), len(text))))
        records.append((10, (0, 0)))
        records = records[:max_length]
        ids = torch.tensor([[token for token, _ in records]], dtype=torch.long)
        encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor(
                [[offset for _, offset in records]], dtype=torch.long
            )
        return encoded


@pytest.mark.parametrize(
    "input_text, prompt",
    [("", PROMPT), ("  context  ", "### Instruction:\nReply.\n\n### Input:\ncontext\n\n### Response:\n")],
)
@pytest.mark.parametrize("answer", ["ANSWER", "你好"])
def test_only_answer_suffix_is_supervised_without_changing_inputs(input_text, prompt, answer):
    sample = {**SAMPLE, "input": input_text, "output": f" {answer} "}
    tokenizer = OffsetTokenizer()
    full = SFTDataset([sample], tokenizer)[0]
    masked = SFTDataset([sample], tokenizer, answer_only=True)[0]
    first_answer = 1 + len(prompt)  # One BOS token precedes character tokens.

    assert set(masked) == {"input_ids", "attention_mask", "labels"}
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(masked[key], full[key])
    assert torch.equal(full["labels"], full["input_ids"])
    assert masked["labels"].dtype == torch.long
    assert masked["labels"].shape == full["labels"].shape
    assert masked["labels"].data_ptr() != masked["input_ids"].data_ptr()
    assert (masked["labels"][:first_answer] == -100).all()
    expected = [ord(char) % 7 + 1 for char in answer + "\n\n"] + [9]
    assert masked["labels"][first_answer:-1].tolist() == expected
    assert masked["labels"][-1] == -100  # Zero-width trailing special token.
    assert masked["labels"][first_answer] == masked["input_ids"][first_answer]
    assert (masked["labels"][1:] != -100).sum() == len(expected)


def test_token_overlapping_prompt_and_answer_is_supervised():
    item = SFTDataset([SAMPLE], OffsetTokenizer(merge_boundary=True), answer_only=True)[0]
    # The merged token covers the final prompt newline and the first answer A.
    boundary_index = len(PROMPT)
    assert (item["labels"][:boundary_index] == -100).all()
    assert item["labels"][boundary_index] == 11
    assert item["input_ids"][boundary_index] == 11
    assert item["labels"][boundary_index + 1] == ord("N") % 7 + 1


@pytest.mark.parametrize("answer_chars_kept", [0, 1, 3])
def test_truncation_at_or_inside_answer(answer_chars_kept):
    limit = 1 + len(PROMPT) + answer_chars_kept
    item = SFTDataset([SAMPLE], OffsetTokenizer(), max_length=limit, answer_only=True)[0]

    assert len(item["input_ids"]) == limit
    assert item["attention_mask"].tolist() == [1] * limit
    assert (item["labels"][1:] != -100).sum() == answer_chars_kept
    expected = [ord(char) % 7 + 1 for char in "ANSWER"[:answer_chars_kept]]
    assert item["labels"][item["labels"] != -100].tolist() == expected
    assert 9 not in item["input_ids"].tolist()  # EOS was truncated too.


def test_no_eos_tokenizer_still_supervises_answer():
    item = SFTDataset([SAMPLE], OffsetTokenizer(eos=False), answer_only=True)[0]
    expected = [ord(char) % 7 + 1 for char in "ANSWER\n\n"]
    assert item["labels"][item["labels"] != -100].tolist() == expected


def test_dynamic_padding_preserves_answer_targets_and_source_items():
    samples = [SAMPLE, {**SAMPLE, "output": "A"}]
    features = list(SFTDataset(samples, OffsetTokenizer(), answer_only=True))
    before = [{key: value.clone() for key, value in item.items()} for item in features]
    batch = SFTDataCollator(pad_token_id=0)(features)

    assert (batch["labels"][:, 1:] != -100).sum() == 9 + 4
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["input_ids"][batch["attention_mask"] == 0] == 0).all()
    for row, item in enumerate(before):
        length = len(item["input_ids"])
        for key, value in item.items():
            assert torch.equal(features[row][key], value)
            assert torch.equal(batch[key][row, :length], value)


class FixedLogitsModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        logits = torch.arange(16, dtype=torch.float32) / 10
        expanded = logits.expand(*input_ids.shape, 16)
        loss = torch.nn.functional.cross_entropy(
            expanded[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_answer_only_dataset_to_evaluation_uses_only_shifted_answer_targets(batch_size):
    samples = [SAMPLE, {**SAMPLE, "output": "A"}]
    dataset = SFTDataset(samples, OffsetTokenizer(), answer_only=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=SFTDataCollator(0))
    result = evaluation_loss(FixedLogitsModel(), loader, device="cpu")

    targets = [ord(char) % 7 + 1 for char in "ANSWER\n\n"] + [9]
    targets += [ord(char) % 7 + 1 for char in "A\n\n"] + [9]
    logits = torch.arange(16, dtype=torch.float32) / 10
    expected = torch.logsumexp(logits, dim=0) - logits[targets].mean()
    assert result == pytest.approx(expected.item())


def test_all_answers_truncated_raises_and_restores_model_mode():
    dataset = SFTDataset([SAMPLE], OffsetTokenizer(), max_length=8, answer_only=True)
    loader = DataLoader(dataset, collate_fn=SFTDataCollator(0))
    model = FixedLogitsModel()
    model.train()

    with pytest.raises(ValueError, match="No valid prediction tokens"):
        evaluation_loss(model, loader, device="cpu")

    assert model.training
