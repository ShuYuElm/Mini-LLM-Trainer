import torch
from pathlib import Path

from transformers import AutoModelForCausalLM

from src.lora import inject_lora
from src.utils import save_lora_adapter

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model"

full_path = ROOT / "checkpoint" / "lora_smoke" / "epoch_2.pt"
adapter_path = ROOT / "checkpoint" / "lora_smoke" / "adapter_epoch_2.pt"

def main():
    if adapter_path.exists():
        raise FileExistsError(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
    )

    model.requires_grad_(False)
    assert inject_lora(model, rank=8, alpha=16) == 56

    checkpoint = torch.load(
        full_path, map_location="cpu", weights_only=True
    )

    model.load_state_dict(checkpoint["model"])

    del checkpoint

    save_lora_adapter(adapter_path, model, MODEL_NAME, 8, 16, ("q_proj", "v_proj"))

if __name__ == "__main__":
    main()

