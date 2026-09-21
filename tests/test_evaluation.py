import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.collator import SFTDataCollator
from src.evaluation import compute_perplexity, evaluation_loss, generate_response


class TinyCausalModel(torch.nn.Module):
    """Every position predicts logits [0, 1, 2, 3], with a real causal CE loss."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[0.0, 1.0, 2.0, 3.0]] * 4)
        )
        self.observations = []

    def forward(self, input_ids, attention_mask, labels):
        inputs = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        logits = inputs @ self.weight
        self.observations.append(
            {
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
                "logits_require_grad": logits.requires_grad,
                "logits_dtype": logits.dtype,
                "devices": [x.device for x in (input_ids, attention_mask, labels)],
                "labels": labels.detach().clone(),
            }
        )
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, 4),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


class RecordingTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self):
        self.prompt = None
        self.call_kwargs = None
        self.decoded_ids = None
        self.decode_kwargs = None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        self.decoded_ids = token_ids.detach().cpu().clone()
        self.decode_kwargs = kwargs
        return "  decoded response  "


class RecordingGenerationModel(torch.nn.Module):
    def __init__(self, should_fail=False):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.tensor(1.0))
        self.should_fail = should_fail
        self.generate_kwargs = None
        self.training_during_generate = None
        self.grad_enabled_during_generate = None
        self.inference_mode_during_generate = None

    def generate(self, **kwargs):
        self.generate_kwargs = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in kwargs.items()
        }
        self.training_during_generate = self.training
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        self.inference_mode_during_generate = torch.is_inference_mode_enabled()

        if self.should_fail:
            raise RuntimeError("generation failed")

        continuation = torch.tensor(
            [[70, 71, self.generate_kwargs["eos_token_id"]]],
            dtype=torch.long,
            device=kwargs["input_ids"].device,
        )
        return torch.cat((kwargs["input_ids"], continuation), dim=1)


@pytest.mark.parametrize(
    ("loss", "expected_perplexity"),
    [(0.0, 1.0), (math.log(4.0), 4.0), (math.log(10.0), 10.0)],
)
def test_compute_perplexity_matches_exponential_definition(
    loss, expected_perplexity
):
    result = compute_perplexity(loss)

    assert isinstance(result, float)
    assert result == pytest.approx(expected_perplexity)


def make_feature(input_ids, labels):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_batch(labels):
    input_ids = [label if label >= 0 else 0 for label in labels]
    return SFTDataCollator(pad_token_id=0)([make_feature(input_ids, labels)])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_loss_is_token_weighted_and_independent_of_batch_grouping(batch_size):
    features = [
        make_feature([1, 0], [-100, 0]),
        make_feature([2, 3, 3, 3], [2, 3, 3, 3]),
    ]
    dataloader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SFTDataCollator(pad_token_id=0),
    )
    model = TinyCausalModel()

    result = evaluation_loss(model, dataloader, device="cpu")

    # Four targets: class 0 once, class 3 three times. The first tokens and
    # padding are not targets. Averaging the two sample losses would be wrong.
    log_partition = math.log(sum(math.exp(x) for x in [0, 1, 2, 3]))
    expected_loss = log_partition - 9 / 4
    assert isinstance(result, float)
    assert result == pytest.approx(expected_loss)
    assert len(model.observations) == (2 if batch_size == 1 else 1)


@pytest.mark.parametrize("initial_training", [False, True])
def test_evaluation_preserves_mode_parameters_gradients_and_input(initial_training):
    model = TinyCausalModel()
    model.train(initial_training)
    model.weight.grad = torch.full_like(model.weight, 7.0)
    weight_before = model.weight.detach().clone()
    gradient_before = model.weight.grad.clone()
    batch = make_batch([2, 0, 3])
    batch_before = {name: tensor.clone() for name, tensor in batch.items()}

    # The current API intentionally disables AMP on CPU, even when requested.
    with torch.enable_grad():
        evaluation_loss(model, [batch], device=torch.device("cpu"), use_amp=True)
        assert torch.is_grad_enabled()

    assert model.training is initial_training
    torch.testing.assert_close(model.weight, weight_before, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, gradient_before, rtol=0, atol=0)
    observation = model.observations[0]
    assert observation["training"] is False
    assert observation["grad_enabled"] is False
    assert observation["logits_require_grad"] is False
    assert observation["logits_dtype"] == torch.float32
    assert observation["devices"] == [torch.device("cpu")] * 3
    assert torch.equal(observation["labels"], batch_before["labels"])
    for name, tensor in batch_before.items():
        assert torch.equal(batch[name], tensor)


@pytest.mark.parametrize(
    "invalid_labels", [[-100, -100, -100], [2, -100, -100], [2]]
)
def test_batches_without_prediction_targets_are_skipped(invalid_labels):
    model = TinyCausalModel()
    invalid_batch = make_batch(invalid_labels)
    valid_batch = make_batch([2, 3])

    result = evaluation_loss(
        model, [invalid_batch, valid_batch, invalid_batch], device="cpu"
    )

    expected_loss = math.log(sum(math.exp(x) for x in [0, 1, 2, 3])) - 3
    assert result == pytest.approx(expected_loss)
    assert len(model.observations) == 1
    assert model.weight.grad is None


@pytest.mark.parametrize("initial_training", [False, True])
@pytest.mark.parametrize("empty_loader", [False, True])
def test_no_valid_targets_raises_and_restores_mode(initial_training, empty_loader):
    model = TinyCausalModel()
    model.train(initial_training)
    dataloader = [] if empty_loader else [make_batch([2, -100])]

    with pytest.raises(ValueError, match="No valid prediction tokens"):
        evaluation_loss(model, dataloader, device="cpu")

    assert model.training is initial_training
    assert model.observations == []


@pytest.mark.parametrize("initial_training", [False, True])
def test_forward_error_restores_mode_and_grad_context(initial_training):
    class FailingModel(TinyCausalModel):
        def forward(self, **batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            raise RuntimeError("intentional forward failure")

    model = FailingModel()
    model.train(initial_training)

    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            evaluation_loss(model, [make_batch([2, 3])], device="cpu")
        assert torch.is_grad_enabled()

    assert model.training is initial_training


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with bf16 support is required",
)
@pytest.mark.parametrize("use_amp", [False, True])
def test_cuda_device_transfer_and_actual_autocast_dtype(use_amp):
    device = torch.device("cuda", torch.cuda.current_device())
    model = TinyCausalModel().to(device)
    batch = make_batch([2, 3])

    result = evaluation_loss(model, [batch], device=device, use_amp=use_amp)

    assert math.isfinite(result)
    observation = model.observations[0]
    assert observation["devices"] == [device] * 3
    assert observation["logits_dtype"] == (
        torch.bfloat16 if use_amp else torch.float32
    )
    assert not observation["logits_require_grad"]
    assert model.weight.grad is None


@pytest.mark.parametrize(
    ("input_text", "expected_prompt"),
    [
        (
            "",
            "### Instruction:\nExplain gradient accumulation.\n\n"
            "### Response:\n",
        ),
        (
            "  a concrete example  ",
            "### Instruction:\nExplain gradient accumulation.\n\n"
            "### Input:\na concrete example\n\n"
            "### Response:\n",
        ),
    ],
)
@pytest.mark.parametrize("initial_training", [False, True])
def test_generate_response_formats_prompt_and_decodes_only_new_tokens(
    input_text, expected_prompt, initial_training
):
    model = RecordingGenerationModel()
    model.train(initial_training)
    tokenizer = RecordingTokenizer()

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        instruction="  Explain gradient accumulation.  ",
        input_text=input_text,
        device="cpu",
        max_input_length=128,
        max_new_tokens=17,
        use_amp=True,
    )

    assert response == "decoded response"
    assert tokenizer.prompt == expected_prompt
    assert tokenizer.call_kwargs == {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 128,
        "padding": False,
    }
    assert model.training is initial_training
    assert model.training_during_generate is False
    assert model.grad_enabled_during_generate is False
    assert model.inference_mode_during_generate is True
    assert model.generate_kwargs["max_new_tokens"] == 17
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert model.generate_kwargs["pad_token_id"] == tokenizer.pad_token_id
    assert model.generate_kwargs["input_ids"].device == torch.device("cpu")
    assert model.generate_kwargs["attention_mask"].device == torch.device("cpu")
    assert torch.equal(tokenizer.decoded_ids, torch.tensor([70, 71, 99]))
    assert tokenizer.decode_kwargs == {"skip_special_tokens": True}


@pytest.mark.parametrize("initial_training", [False, True])
def test_generate_error_restores_model_mode_and_grad_context(initial_training):
    model = RecordingGenerationModel(should_fail=True)
    model.train(initial_training)
    tokenizer = RecordingTokenizer()

    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="generation failed"):
            generate_response(
                model=model,
                tokenizer=tokenizer,
                instruction="test",
                device="cpu",
            )
        assert torch.is_grad_enabled()

    assert model.training is initial_training
    assert model.training_during_generate is False
    assert model.grad_enabled_during_generate is False
    assert model.inference_mode_during_generate is True
