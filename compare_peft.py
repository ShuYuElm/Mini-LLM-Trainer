"""Learning exercise: compare manual LoRA with PEFT on one small projection.

The setup and parameter inspection are provided. Complete the TODOs in main
before treating this script as an equivalence check.
"""

from copy import deepcopy

import torch
from torch import nn
from peft import LoraConfig, get_peft_model

from src.lora import LoRALinear


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x):
        return self.proj(x)


def describe_parameters(label, model):
    print(f"\n{label}:")
    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape), "trainable:", parameter.requires_grad)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", count)
    assert count == 2 * (4 + 3)


def main():
    torch.manual_seed(0)
    base = TinyModel().to(device="cpu", dtype=torch.float32)

    # Independent copies with identical base weights and biases.
    manual_model = deepcopy(base)
    manual_model.proj = LoRALinear(manual_model.proj, rank=2, alpha=4)

    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["proj"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        use_dora=False,
        init_lora_weights=True,
    )
    peft_model = get_peft_model(deepcopy(base), config)

    describe_parameters("Manual LoRA", manual_model)
    describe_parameters("PEFT LoRA", peft_model)

    manual_layer = manual_model.proj
    peft_layer = peft_model.get_base_model().proj
    x = torch.randn(2, 5, 4)
    manual_model.eval()
    peft_model.eval()

    print("input:", x.shape, x.dtype, x.device)
    print("manual scaling:", manual_layer.scaling)
    print("PEFT scaling:", peft_layer.scaling["default"])

    with torch.no_grad():
        manual_layer.lora_B.weight.normal_(mean=0.0, std=0.1)

        peft_layer.lora_A["default"].weight.copy_(
            manual_layer.lora_A.weight
        )

        peft_layer.lora_B["default"].weight.copy_(
            manual_layer.lora_B.weight
        )

    with torch.no_grad():
        manual_output = manual_model(x)
        peft_output = peft_model(x)
        base_output = base(x)

    assert manual_output.shape == (2, 5, 3)
    assert peft_output.shape == (2, 5, 3)

    max_difference = (
        manual_output - peft_output
    ).abs().max().item()

    print("output shape:", manual_output.shape)
    print("max output difference:", max_difference)

    torch.testing.assert_close(
        manual_output,
        peft_output,
        rtol=1e-5,
        atol=1e-6,
    )

    update_size = (
        manual_output - base_output
    ).abs().max().item()

    print("max difference from base:", update_size)

    assert not torch.allclose(manual_output, base_output)

    print("Forward comparison passed.")

    manual_model.zero_grad()
    peft_model.zero_grad()

    target = torch.randn_like(manual_output)

    manual_prediction = manual_model(x)
    peft_prediction = peft_model(x)

    manual_loss = nn.functional.mse_loss(manual_prediction, target)
    peft_loss =nn.functional.mse_loss(peft_prediction, target)

    manual_loss.backward()
    peft_loss.backward()

    print("manual loss:", manual_loss.item())
    print("PEFT loss", peft_loss.item())

    manual_A_grad = manual_layer.lora_A.weight.grad
    peft_A_grad = peft_layer.lora_A["default"].weight.grad

    assert manual_A_grad is not None
    assert peft_A_grad is not None

    print(
        "max A gradient difference",
        (manual_A_grad - peft_A_grad).abs().max().item()
    )

    torch.testing.assert_close(
        manual_A_grad,
        peft_A_grad,
        rtol=1e-6,
        atol=1e-6
    )

    manual_B_grad = manual_layer.lora_B.weight.grad
    peft_B_grad = peft_layer.lora_B["default"].weight.grad

    assert manual_B_grad is not None
    assert peft_B_grad is not None

    print(
        "max B gradient difference:",
        (manual_B_grad - peft_B_grad).abs().max().item()
    )

    torch.testing.assert_close(
        manual_B_grad,
        peft_B_grad,
        rtol=1e-5,
        atol=1e-6
    )

    for parameter in manual_layer.base_layer.parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None

    for parameter in peft_layer.base_layer.parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None

    print("Gradient comparison passed.")

    manual_optimizer = torch.optim.SGD(
        (p for p in manual_model.parameters() if p.requires_grad),
        lr=0.01,
    )

    peft_optimizer = torch.optim.SGD(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=0.01,
    )

    manual_before = {
        name: parameter.detach().clone()
        for name, parameter in manual_model.named_parameters()
    }

    peft_before = {
        name: parameter.detach().clone()
        for name, parameter in peft_model.named_parameters()
    }

    manual_optimizer.step()
    peft_optimizer.step()

    for model, before in (
        (manual_model, manual_before),
        (peft_model, peft_before),
    ):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert not torch.equal(parameter, before[name]), name
            else:
                torch.testing.assert_close(
                    parameter,
                    before[name],
                    rtol=0,
                    atol=0,
                )

    torch.testing.assert_close(
        manual_layer.lora_A.weight,
        peft_layer.lora_A["default"].weight,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        manual_layer.lora_B.weight,
        peft_layer.lora_B["default"].weight,
        rtol=1e-5,
        atol=1e-6,
    )

    with torch.no_grad():
        manual_after = manual_model(x)
        peft_after = peft_model(x)

    torch.testing.assert_close(
        manual_after,
        peft_after,
        rtol=1e-5,
        atol=1e-6,
    )

    print(
        "max output difference after step:",
        (manual_after - peft_after).abs().max().item(),
    )
    print("Optimizer update comparison passed.")


if __name__ == "__main__":
    main()
