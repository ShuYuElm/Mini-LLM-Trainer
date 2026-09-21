import torch
from pathlib import Path

from transformers import AutoModelForCausalLM

from src.lora import inject_lora

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        dtype=torch.bfloat16,
    )

    model.to(device)

    model.requires_grad_(False)

    changes = inject_lora(model, rank=8, alpha=16)

    print(changes)
    assert changes == 56

    trainable = {
        name: p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    print(trainable)
    print("total:", sum(trainable.values()))



if __name__ == "__main__":
    main()