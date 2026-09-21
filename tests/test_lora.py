import pytest
import torch
from torch import nn

from src.lora import LoRALinear, inject_lora


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 6, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(6, 4, bias=False)


class TinyQwenLikeModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Module() for _ in range(num_layers)]
        )
        for block in self.model.layers:
            block.self_attn = TinyAttention()
        self.lm_head = nn.Linear(4, 4)


@pytest.mark.parametrize("input_shape", [(5, 4), (2, 5, 4)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_initial_output_matches_base_linear(input_shape, with_bias):
    torch.manual_seed(0)
    base = nn.Linear(4, 3, bias=with_bias)
    x = torch.randn(*input_shape)
    expected = base(x)

    layer = LoRALinear(base, rank=2, alpha=4)
    actual = layer(x)

    assert actual.shape == (*input_shape[:-1], 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(layer.lora_B.weight) == 0


def test_shapes_scaling_and_trainable_parameter_count():
    base = nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=4)

    assert layer.lora_A.weight.shape == (2, 4)
    assert layer.lora_B.weight.shape == (3, 2)
    assert layer.scaling == 2
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert {
        name for name, parameter in layer.named_parameters() if parameter.requires_grad
    } == {"lora_A.weight", "lora_B.weight"}
    assert sum(parameter.numel() for parameter in layer.parameters()
               if parameter.requires_grad) == 14


def test_nonzero_lora_update_uses_original_input_and_alpha_over_rank():
    base = nn.Linear(2, 1, bias=False)
    layer = LoRALinear(base, rank=1, alpha=2)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[10.0, 20.0]]))
        layer.lora_A.weight.copy_(torch.tensor([[1.0, 2.0]]))
        layer.lora_B.weight.copy_(torch.tensor([[3.0]]))

    # base([4, 5]) = 140; A([4, 5]) = 14; B(A(x)) = 42;
    # alpha / rank = 2, so the final result is 140 + 2 * 42 = 224.
    output = layer(torch.tensor([[4.0, 5.0]]))

    assert output.shape == (1, 1)
    assert output.item() == pytest.approx(224.0)


def test_first_backward_freezes_base_but_reaches_lora_and_input():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(5, 4, requires_grad=True)

    layer(x).square().sum().backward()

    assert all(parameter.grad is None for parameter in base.parameters())
    assert layer.lora_A.weight.grad is not None
    assert torch.count_nonzero(layer.lora_A.weight.grad) == 0
    assert layer.lora_B.weight.grad is not None
    assert torch.count_nonzero(layer.lora_B.weight.grad) > 0
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) > 0


def test_optimizer_updates_adapter_without_changing_base():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(5, 4)
    base_before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    initial_output = layer(x).detach().clone()
    optimizer = torch.optim.SGD(
        (parameter for parameter in layer.parameters() if parameter.requires_grad),
        lr=0.01,
    )

    layer(x).square().sum().backward()
    optimizer.step()

    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    assert torch.count_nonzero(layer.lora_B.weight) > 0
    assert not torch.equal(layer(x), initial_output)

    optimizer.zero_grad()
    layer(x).square().sum().backward()
    assert torch.count_nonzero(layer.lora_A.weight.grad) > 0


@pytest.mark.parametrize("rank", [0, -1])
def test_rank_must_be_positive(rank):
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(4, 3), rank=rank, alpha=4)


def test_base_layer_must_be_linear():
    with pytest.raises(TypeError, match="base_layer"):
        LoRALinear(nn.ReLU(), rank=2, alpha=4)


def test_lora_layers_match_base_dtype():
    base = nn.Linear(4, 3).to(dtype=torch.float64)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(2, 4, dtype=torch.float64)

    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    assert layer.lora_A.weight.dtype == base.weight.dtype
    assert layer.lora_B.weight.dtype == base.weight.dtype


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA with bf16 support is required",
)
def test_lora_layers_match_cuda_bf16_base():
    device = torch.device("cuda")
    base = nn.Linear(4, 3).to(device=device, dtype=torch.bfloat16)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(2, 4, device=device, dtype=torch.bfloat16)

    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    for adapter in (layer.lora_A, layer.lora_B):
        assert adapter.weight.device == base.weight.device
        assert adapter.weight.dtype == base.weight.dtype


def test_inject_lora_wraps_only_q_and_v_in_every_layer():
    model = TinyQwenLikeModel(num_layers=2)
    model.requires_grad_(False)
    original_modules = [
        {
            name: getattr(block.self_attn, name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        for block in model.model.layers
    ]
    x = torch.randn(2, 4)
    original_outputs = [
        {name: modules[name](x) for name in ("q_proj", "v_proj")}
        for modules in original_modules
    ]

    replaced = inject_lora(model, rank=2, alpha=4)

    assert replaced == 4
    for index, block in enumerate(model.model.layers):
        attention = block.self_attn
        originals = original_modules[index]
        for name in ("q_proj", "v_proj"):
            wrapped = getattr(attention, name)
            assert isinstance(wrapped, LoRALinear)
            assert wrapped.base_layer is originals[name]
            torch.testing.assert_close(
                wrapped(x), original_outputs[index][name], rtol=0, atol=0
            )
        for name in ("k_proj", "o_proj"):
            assert getattr(attention, name) is originals[name]

    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_names = {
        f"model.layers.{index}.self_attn.{target}.lora_{adapter}.weight"
        for index in range(2)
        for target in ("q_proj", "v_proj")
        for adapter in ("A", "B")
    }
    assert set(trainable) == expected_names
    # Per layer: q has 2*(4+6)=20 and v has 2*(4+4)=16 parameters.
    assert sum(trainable.values()) == 2 * (20 + 16)


def test_inject_lora_respects_custom_target_names():
    model = TinyQwenLikeModel(num_layers=2)
    model.requires_grad_(False)
    original_v = [block.self_attn.v_proj for block in model.model.layers]

    replaced = inject_lora(model, rank=2, alpha=4, target_names=("q_proj",))

    assert replaced == 2
    for block, v_proj in zip(model.model.layers, original_v):
        assert isinstance(block.self_attn.q_proj, LoRALinear)
        assert block.self_attn.v_proj is v_proj
    assert sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad) == 2 * 20


def test_inject_lora_rejects_already_wrapped_projection():
    model = TinyQwenLikeModel(num_layers=2)
    model.requires_grad_(False)
    inject_lora(model, rank=2, alpha=4)
    first_q_proj = model.model.layers[0].self_attn.q_proj

    with pytest.raises(TypeError, match="base_layer"):
        inject_lora(model, rank=2, alpha=4)

    assert model.model.layers[0].self_attn.q_proj is first_q_proj
