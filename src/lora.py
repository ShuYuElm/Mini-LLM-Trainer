import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.device = base_layer.weight.device
        self.dtype = base_layer.weight.dtype

        self.base_layer.requires_grad_(False)
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(
            base_layer.in_features,
            rank,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        self.lora_B = nn.Linear(
            rank,
            base_layer.out_features,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_y = self.base_layer(x)

        lora_y = self.lora_B(self.lora_A(x)) * self.scaling

        return base_y + lora_y


def inject_lora(model, rank=8, alpha=16, target_names=("q_proj", "v_proj")):
    """替换 Qwen attention 中指定的 Linear，返回替换数量。"""
    changes = 0

    for block in model.model.layers:
        attention = block.self_attn
        for target in target_names:
            original_projection = getattr(attention, target)

            new_projection = LoRALinear(original_projection, rank, alpha)

            setattr(attention, target, new_projection)

            changes += 1

    return changes

