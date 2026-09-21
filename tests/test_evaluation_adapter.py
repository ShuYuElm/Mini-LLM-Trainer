"""Check the evaluation loader's metadata contract with tiny CPU models."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import evaluate
from src.lora import inject_lora
from src.utils import save_lora_adapter


def make_model():
    attention = nn.Module()
    attention.q_proj = nn.Linear(4, 3, bias=False)
    block = nn.Module()
    block.self_attn = attention
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([block])
    return model


def test_wrong_base_model_is_rejected_before_freezing_or_injection(tmp_path, monkeypatch):
    path = tmp_path / "wrong_base.pt"
    torch.save({"base_model_name": "different/base"}, path)
    model = make_model()
    projection = model.model.layers[0].self_attn.q_proj
    before = projection.weight.detach().clone()
    inject = Mock(side_effect=AssertionError("Must validate identity before injection"))
    load = Mock(side_effect=AssertionError("Must validate identity before copying"))
    monkeypatch.setattr(evaluate, "inject_lora", inject)
    monkeypatch.setattr(evaluate, "load_lora_adapter", load)

    with pytest.raises(ValueError, match="base model does not match"):
        evaluate.load_adapter_for_evaluation(model, path, "expected/base")

    inject.assert_not_called()
    load.assert_not_called()
    assert model.model.layers[0].self_attn.q_proj is projection
    assert projection.weight.requires_grad
    assert torch.equal(projection.weight, before)


@pytest.mark.parametrize("rank, alpha", [(1, 3), (2, 10)])
def test_evaluation_loads_saved_rank_alpha_and_weights(tmp_path, rank, alpha):
    source = make_model()
    destination = make_model()
    destination.load_state_dict(source.state_dict())
    source.requires_grad_(False)
    inject_lora(source, rank=rank, alpha=alpha, target_names=["q_proj"])
    trained_projection = source.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        trained_projection.lora_A.weight.fill_(0.25)
        trained_projection.lora_B.weight.fill_(0.5)
    path = tmp_path / "adapter.pt"
    save_lora_adapter(path, source, "toy/base", rank, alpha, ["q_proj"])

    metadata = evaluate.load_adapter_for_evaluation(destination, path, "toy/base")

    restored = destination.model.layers[0].self_attn.q_proj
    assert metadata == {
        "base_model_name": "toy/base", "rank": rank, "alpha": alpha,
        "target_names": ["q_proj"],
    }
    assert restored.scaling == alpha / rank
    assert restored.lora_A.weight.shape == (rank, 4)
    assert restored.lora_B.weight.shape == (3, rank)
    assert not restored.base_layer.weight.requires_grad
    assert torch.equal(restored.base_layer.weight, trained_projection.base_layer.weight)
    assert torch.equal(restored.lora_A.weight, trained_projection.lora_A.weight)
    assert torch.equal(restored.lora_B.weight, trained_projection.lora_B.weight)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(restored(inputs), trained_projection(inputs), rtol=0, atol=0)
