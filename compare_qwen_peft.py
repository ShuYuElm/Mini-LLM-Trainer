import torch
import gc

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.utils import load_lora_adapter
from verify_lora_adapter import build_lora_model, get_last_token_logits

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model"

ADAPTER_PATH = (
    Path(__file__).resolve().parent
    / "checkpoint"
    / "lora_auto_save_smoke"
    / "adapter_epoch_2.pt"
)

def build_peft_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
    )

    model.to(device)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        use_dora=False,
    )

    model = get_peft_model(
        model,
        config,
        autocast_adapter_dtype=False,
    )

    return model

def load_manual_adapter_into_peft(model, adapter_path):
    payload = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=True,
    )

    config = model.peft_config["default"]

    if payload["base_model_name"] != MODEL_NAME:
        raise ValueError("base model does not match")

    if payload["rank"] != config.r:
        raise ValueError("LoRA rank does not match")

    if payload["alpha"] != config.lora_alpha:
        raise ValueError("LoRA alpha does not match")

    if set(payload["target_names"]) != set(config.target_modules):
        raise ValueError("LoRA target modules do not match")

    mapped = {}

    for name, tensor in payload["adapter"].items():
        peft_name = "base_model.model." + name
        peft_name = peft_name.replace(
            ".lora_A.weight",
            ".lora_A.default.weight",
        )
        peft_name = peft_name.replace(
            ".lora_B.weight",
            ".lora_B.default.weight",
        )
        mapped[peft_name] = tensor

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith((
            ".lora_A.default.weight",
            ".lora_B.default.weight",
        ))
    }

    if not current:
        raise ValueError("model has no default LoRA adapter parameters")

    if set(mapped) != set(current):
        raise ValueError("adapter parameter names do not match")

    for name, parameter in current.items():
        if mapped[name].shape != parameter.shape:
            raise ValueError(f"adapter shape does not match: {name}")

    with torch.no_grad():
        for name, parameter in current.items():
            source = mapped[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(source)

            torch.testing.assert_close(
                parameter,
                source,
                rtol=0,
                atol=0,
            )

    return len(current)

    

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_peft_model(device)

    trainable_count = 0
    trainable_tensor_count = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        print(
            name,
            tuple(parameter.shape),
            parameter.dtype,
            parameter.device,
        )

        trainable_count +=  parameter.numel()
        trainable_tensor_count += 1

        assert name.endswith((
            ".lora_A.default.weight",
            ".lora_B.default.weight",
        ))
        assert parameter.dtype == torch.bfloat16
        assert parameter.device.type == device.type

    print("trainable tensors:", trainable_tensor_count)
    print("trainable parameters:", trainable_count)

    assert trainable_tensor_count == 112
    assert trainable_count == 1_146_880

    print("PEFT Qwen parameter inspection passed")

    loaded_count = load_manual_adapter_into_peft(
        model,
        ADAPTER_PATH,
    )

    print("loaded adapter tensors:", loaded_count)

    assert loaded_count == 112
    print("Manual adapter loading into PEFT passed.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
    )

    prompt= (
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

    model.eval()

    peft_logits = get_last_token_logits(
        model,
        encoded,
        device,
    )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = torch.load(
        ADAPTER_PATH,
        map_location="cpu",
        weights_only=True,
    )

    manual_model = build_lora_model(
        rank=payload["rank"],
        alpha=payload["alpha"],
        target_names=payload["target_names"],
    )

    load_lora_adapter(ADAPTER_PATH, manual_model)

    manual_model.to(device)
    manual_model.eval()

    manual_logits = get_last_token_logits(
        manual_model,
        encoded,
        device,
    )

    print("PEFT logits:", peft_logits.shape, peft_logits.dtype)
    print("Manual logits:", manual_logits.shape, manual_logits.dtype)

    max_difference = (
        manual_logits - peft_logits
    ).abs().max().item()

    print("max logits difference:", max_difference)

    assert torch.isfinite(manual_logits).all()
    assert torch.isfinite(peft_logits).all()

    torch.testing.assert_close(
        manual_logits,
        peft_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    print("Qwen manual vs PEFT logits comparison passed.")

if __name__ == "__main__":
    main()
