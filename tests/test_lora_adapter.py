import pytest
import torch
from torch import nn

from src.lora import LoRALinear
from src.utils import load_lora_adapter, save_lora_adapter


class ToyModelWithAdapter(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        base = nn.Linear(4, 3).to(device=device, dtype=dtype)
        self.projection = LoRALinear(base, rank=2, alpha=4)
        self.extra_trainable = nn.Linear(3, 1).to(device=device, dtype=dtype)


@pytest.mark.parametrize("path_as_string", [False, True])
def test_save_adapter_keeps_only_named_lora_weights_and_metadata(
    tmp_path, path_as_string
):
    model = ToyModelWithAdapter()
    # An adapter weight still belongs in the file if it was temporarily frozen;
    # the unrelated trainable layer does not.
    model.projection.lora_A.weight.requires_grad_(False)
    path = tmp_path / "nested" / "adapter.pt"
    path_argument = str(path) if path_as_string else path

    save_lora_adapter(
        path_argument,
        model,
        base_model_name="toy/base",
        rank=2,
        alpha=4,
        target_names=("q_proj", "v_proj"),
    )

    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "base_model_name", "rank", "alpha", "target_names", "adapter"
    }
    assert payload["base_model_name"] == "toy/base"
    assert payload["rank"] == 2
    assert payload["alpha"] == 4
    assert payload["target_names"] == ["q_proj", "v_proj"]
    assert set(payload["adapter"]) == {
        "projection.lora_A.weight",
        "projection.lora_B.weight",
    }
    for name, saved in payload["adapter"].items():
        expected = dict(model.named_parameters())[name]
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
        assert saved.device.type == "cpu"
        assert not saved.requires_grad


def test_saved_adapter_is_independent_of_later_model_updates(tmp_path):
    model = ToyModelWithAdapter()
    path = tmp_path / "adapter.pt"
    before = model.projection.lora_B.weight.detach().clone()

    save_lora_adapter(path, model, "toy/base", 2, 4, ("q_proj",))
    with torch.no_grad():
        model.projection.lora_B.weight.add_(5)

    saved = torch.load(path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        saved["adapter"]["projection.lora_B.weight"], before, rtol=0, atol=0
    )


def test_save_adapter_rejects_model_without_lora_parameters(tmp_path):
    path = tmp_path / "empty_adapter.pt"

    with pytest.raises(ValueError, match="no LoRA adapter parameters"):
        save_lora_adapter(path, nn.Linear(4, 3), "toy/base", 2, 4, ("q_proj",))

    assert not path.exists()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with bf16 support is required",
)
def test_save_cuda_bf16_adapter_as_cpu_tensors(tmp_path):
    model = ToyModelWithAdapter(device="cuda", dtype=torch.bfloat16)
    path = tmp_path / "adapter.pt"

    save_lora_adapter(path, model, "toy/base", 2, 4, ("q_proj",))

    saved = torch.load(path, map_location="cpu", weights_only=True)["adapter"]
    assert set(saved) == {"projection.lora_A.weight", "projection.lora_B.weight"}
    for tensor in saved.values():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.bfloat16


def test_adapter_round_trip_restores_weights_outputs_and_metadata(tmp_path):
    torch.manual_seed(0)
    trained = ToyModelWithAdapter()
    with torch.no_grad():
        trained.projection.lora_A.weight.fill_(0.25)
        trained.projection.lora_B.weight.fill_(0.5)
    x = torch.randn(2, 4)
    expected_output = trained.projection(x).detach().clone()
    path = tmp_path / "adapter.pt"
    save_lora_adapter(path, trained, "toy/base", 2, 4, ("q_proj",))

    restored = ToyModelWithAdapter()
    restored.projection.base_layer.load_state_dict(
        trained.projection.base_layer.state_dict()
    )
    assert not torch.equal(
        restored.projection.lora_B.weight, trained.projection.lora_B.weight
    )

    metadata = load_lora_adapter(path, restored)

    assert metadata == {
        "base_model_name": "toy/base",
        "rank": 2,
        "alpha": 4,
        "target_names": ["q_proj"],
    }
    for name in ("lora_A", "lora_B"):
        torch.testing.assert_close(
            getattr(restored.projection, name).weight,
            getattr(trained.projection, name).weight,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        restored.projection(x), expected_output, rtol=0, atol=0
    )


@pytest.mark.parametrize("mismatch", ["missing_key", "wrong_shape"])
def test_bad_adapter_rejected_before_changing_any_parameter(tmp_path, mismatch):
    source = ToyModelWithAdapter()
    path = tmp_path / "adapter.pt"
    save_lora_adapter(path, source, "toy/base", 2, 4, ("q_proj",))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mismatch == "missing_key":
        payload["adapter"].pop("projection.lora_B.weight")
    else:
        payload["adapter"]["projection.lora_B.weight"] = torch.zeros(3, 1)
    torch.save(payload, path)

    destination = ToyModelWithAdapter()
    before = {
        name: parameter.detach().clone()
        for name, parameter in destination.named_parameters()
        if name.endswith((".lora_A.weight", ".lora_B.weight"))
    }

    with pytest.raises(ValueError, match="adapter"):
        load_lora_adapter(path, destination)

    for name, parameter in destination.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with bf16 support is required",
)
def test_load_cpu_adapter_into_cuda_bf16_model(tmp_path):
    source = ToyModelWithAdapter()
    with torch.no_grad():
        source.projection.lora_A.weight.fill_(0.25)
        source.projection.lora_B.weight.fill_(0.5)
    path = tmp_path / "adapter.pt"
    save_lora_adapter(path, source, "toy/base", 2, 4, ("q_proj",))
    destination = ToyModelWithAdapter(device="cuda", dtype=torch.bfloat16)

    load_lora_adapter(path, destination)

    for name in ("lora_A", "lora_B"):
        actual = getattr(destination.projection, name).weight
        expected = getattr(source.projection, name).weight
        assert actual.device.type == "cuda"
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual.cpu(), expected.to(dtype=torch.bfloat16), rtol=0, atol=0
        )
    output = destination.projection(
        torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    )
    assert output.shape == (2, 3)
