"""Run two optimizer updates on long samples without saving checkpoints.

Reads --config and retains the full-run learning-rate schedule.
This checks memory and training mechanics, not fine-tuning quality.
"""

import math
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from train import expected_lora_counts
from src.collator import SFTDataCollator
from src.dataset import SFTDataset
from src.lora import inject_lora
from src.trainer import train_one_epoch
from src.utils import load_json_config, parse_train_args, set_seed, validate_train_config


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    args = parse_train_args()
    config = load_json_config(args.config)
    validate_train_config(config)
    model_cache_dir = (PROJECT_ROOT / config["model_cache_dir"]).resolve()
    data_dir = (PROJECT_ROOT / config["data_dir"]).resolve()

    if config["resume_path"] is not None:
        raise ValueError("Smoke checks start from the base model; resume_path must be null")

    set_seed(config["seed"])
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 8
    assert config["max_length"] == 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=model_cache_dir,
        local_files_only=True,
    )
    splits = load_from_disk(str(data_dir))
    dataset = SFTDataset(
        splits["train"], tokenizer, max_length=config["max_length"]
    )
    assert len(dataset) == 2000

    lengths = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]
    selected_indices = sorted(
        range(len(dataset)), key=lambda i: lengths[i], reverse=True
    )[:16]
    selected_lengths = [lengths[i] for i in selected_indices]
    assert len(selected_indices) == 16
    assert all(length == config["max_length"] for length in selected_lengths)
    print("selected lengths:", selected_lengths, flush=True)
    print("selected sample IDs:", [
        splits["train"][i]["sample_id"] for i in selected_indices
    ], flush=True)

    dataloader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=SFTDataCollator(tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        cache_dir=model_cache_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.to(device)
    model.requires_grad_(False)

    expected_layers, expected_parameters = expected_lora_counts(
        model=model,
        rank=config["lora_rank"],
        target_names=config["lora_target_names"],
    )

    changes = inject_lora(
        model,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        target_names=config["lora_target_names"],
    )
    assert changes == expected_layers

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    
    actual_parameters = sum(p.numel() for p in trainable.values())
    assert actual_parameters == expected_parameters
    print("LoRA rank/alpha:", config["lora_rank"], config["lora_alpha"])
    print("trainable parameters:", actual_parameters)

    for name, parameter in trainable.items():
        assert name.endswith((".lora_A.weight", ".lora_B.weight"))
        assert parameter.dtype == torch.bfloat16
        assert parameter.device.type == device.type

    optimizer = torch.optim.AdamW(
        trainable.values(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Retain the full baseline schedule, even though this run uses a subset.
    full_batches = math.ceil(len(dataset) / config["batch_size"])
    updates_per_epoch = math.ceil(
        full_batches / config["gradient_accumulation_steps"]
    )
    total_steps = updates_per_epoch * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    assert total_steps == 250
    assert warmup_steps == 25
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print("full schedule total/warmup steps:", total_steps, warmup_steps)
    print("initial learning rate:", scheduler.get_last_lr()[0], flush=True)

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainable.items()
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    avg_loss, updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_grad_norm=config["max_grad_norm"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        scheduler=scheduler,
        use_amp=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    # Capture peaks before running additional diagnostic tensor operations.
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    assert updates == 2
    assert math.isfinite(avg_loss)
    for parameter in trainable.values():
        assert torch.isfinite(parameter).all().item()
    changed = [
        name for name, parameter in trainable.items()
        if not torch.equal(parameter, before[name])
    ]
    changed_b = [name for name in changed if name.endswith(".lora_B.weight")]
    assert changed_b, "No LoRA B weights changed after the two updates"

    print("average loss:", avg_loss)
    print("optimizer updates:", updates)
    print("changed adapter tensors:", len(changed))
    print("changed B tensors:", len(changed_b))
    print("learning rate for next update:", scheduler.get_last_lr()[0])
    print(f"training elapsed: {elapsed:.2f} seconds")
    if device.type == "cuda":
        print(f"peak allocated GPU memory: {peak_allocated / 1024**3:.3f} GiB")
        print(f"peak reserved GPU memory: {peak_reserved / 1024**3:.3f} GiB")
    print("Smoke training passed. No checkpoint saved.")


if __name__ == "__main__":
    main()
