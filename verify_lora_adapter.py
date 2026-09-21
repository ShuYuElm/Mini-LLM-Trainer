import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import inject_lora
from src.utils import load_lora_adapter

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model"

FULL_PATH = ROOT / "checkpoint" / "lora_smoke" / "epoch_2.pt"
ADAPTER_PATH = ROOT / "checkpoint" / "lora_smoke" / "adapter_epoch_2.pt"

def build_lora_model(rank, alpha, target_names):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
    )

    model.requires_grad_(False)

    replaces = inject_lora(model, rank, alpha, target_names)

    assert replaces == 56

    return model

def get_last_token_logits(model, encoded, device):

    encoded = encoded.to(device=device)

    with torch.inference_mode():
        output = model(**encoded)
        last_token_logits = output.logits[:, -1, :].float().to("cpu")

    return last_token_logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(
        ADAPTER_PATH,
        map_location="cpu",
        weights_only=True,
    )

    rank = payload["rank"]
    alpha = payload["alpha"]
    target_names = payload["target_names"]

    adapter_model = build_lora_model(rank, alpha, target_names)

    metadata = load_lora_adapter(
        ADAPTER_PATH,
        adapter_model,
    )

    adapter_model.to(device)
    adapter_model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in adapter_model.parameters()
        if parameter.requires_grad
    )

    print("metadata:", metadata)
    print("trainable parameters:", trainable_parameters)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
    )

    prompt = (
        "### Instruction:\n"
        "Explain why the sky appears blue.\n\n"
        "### Response:\n"
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False,
    )

    print("encoded keys:", encoded.keys())
    print("input_ids shape:", encoded["input_ids"].shape)
    print("attention_mask shape:", encoded["attention_mask"].shape)

    adapter_logits = get_last_token_logits(
        adapter_model,
        encoded,
        device,
    )

    print("adapter logits shape:", adapter_logits.shape)
    print("adapter logits dtype:", adapter_logits.dtype)
    print("adapter logits device:", adapter_logits.device)

    del adapter_model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    full_model = build_lora_model(
        rank,
        alpha,
        target_names,
    )

    checkpoint = torch.load(
        FULL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    full_model.load_state_dict(checkpoint["model"])
    del checkpoint

    full_model.to(device)
    full_model.eval()

    full_logits = get_last_token_logits(
        full_model,
        encoded,
        device,
    )

    print("full logits shape:", full_logits.shape)

    max_difference = (
            adapter_logits - full_logits
    ).abs().max().item()

    print("max logits difference:", max_difference)

    torch.testing.assert_close(
        adapter_logits,
        full_logits,
        rtol=0,
        atol=0,
    )

    print("adapter verification passed")





if __name__ == "__main__":
    main()