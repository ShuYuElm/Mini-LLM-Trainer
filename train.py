import math
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from src.collator import SFTDataCollator
from src.dataset import SFTDataset
from src.trainer import train_one_epoch
from src.utils import save_checkpoint, load_checkpoint, save_lora_adapter, set_seed, save_json, load_json_config, parse_train_args, validate_train_config
from src.evaluation import evaluation_loss, compute_perplexity
from src.lora import inject_lora

PROJECT_ROOT = Path(__file__).resolve().parent

def expected_lora_counts(model, rank, target_names):
    expected_layers = 0
    expected_parameters = 0

    for block in model.model.layers:
        for target_name in target_names:
            projection = getattr(block.self_attn, target_name)

            expected_layers += 1
            expected_parameters += rank * (
                projection.in_features + projection.out_features
            )

    return expected_layers, expected_parameters


def main():
    args = parse_train_args()
    config = load_json_config(args.config)
    validate_train_config(config)

    model_cache_dir = (PROJECT_ROOT / config["model_cache_dir"]).resolve()
    data_dir = (PROJECT_ROOT / config["data_dir"]).resolve()
    checkpoint_dir = (PROJECT_ROOT / config["checkpoint_dir"]).resolve()

    resume_path = (
        (PROJECT_ROOT / config["resume_path"]).resolve()
        if config["resume_path"] is not None
        else None
    )

    if resume_path is None and checkpoint_dir.exists():
        raise FileExistsError(
            f"choose a new run directory: {checkpoint_dir}"
        )

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=model_cache_dir,
    )

    splits = load_from_disk(str(data_dir))

    raw_dataset = splits["train"]
    raw_val_dataset = splits["validation"]

    assert len(raw_dataset) == 2000
    assert len(raw_val_dataset) == 200

    print("training samples:", len(raw_dataset))
    print("validation samples:", len(raw_val_dataset))

    dataset = SFTDataset(
        raw_dataset,
        tokenizer,
        max_length=config["max_length"],
    )

    val_dataset = SFTDataset(
        raw_val_dataset,
        tokenizer,
        max_length=config["max_length"],
    )

    collator = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        cache_dir=model_cache_dir,
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
        model=model,
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        target_names=config["lora_target_names"],
    )

    model.train()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert changes == expected_layers, (
        f"替换层数不一致：实际 {changes}，预期 {expected_layers}"
    )
    assert trainable_parameters == expected_parameters, (
        f"可训练参数量不一致：实际 {trainable_parameters}，"
        f"预期 {expected_parameters}"
    )

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert name.endswith((".lora_A.weight", ".lora_B.weight")), (
                f"发现非 LoRA 可训练参数：{name}"
            )

    probe_parameter = model.model.layers[0].self_attn.q_proj.lora_B.weight

    print("total parameters:", total_parameters)
    print("trainable parameters:", trainable_parameters)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    updates_per_epoch = math.ceil(
        len(dataloader) / config["gradient_accumulation_steps"]
    )

    num_training_steps = updates_per_epoch * config["num_epochs"]

    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    start_epoch = 0
    global_step = 0
    save_dir = checkpoint_dir

    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        save_dir = checkpoint_dir / "resumed"

    initial_val_loss = evaluation_loss(
        model=model,
        dataloader=val_dataloader,
        device=device,
        use_amp=True,
    )

    initial_perplexity = compute_perplexity(initial_val_loss)

    print(
        "validation loss before current run:",
        initial_val_loss,
    )
    print(
        "validation perplexity before current run:",
        initial_perplexity,
    )
    print("updates per epoch:", updates_per_epoch)
    print("training steps:", num_training_steps)
    print("warmup steps:", num_warmup_steps)
    print(
        "initial learning rate:",
        optimizer.param_groups[0]["lr"],
    )

    run_config = {
        "model_name": config["model_name"],
        "data_dir": str(data_dir.resolve()),
        "seed": config["seed"],
        "train_samples": len(dataset),
        "validation_samples": len(val_dataset),
        "max_length": config["max_length"],
        "label_policy": "full_sequence",
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "num_epochs": config["num_epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "optimizer": "AdamW",
        "scheduler": "linear_with_warmup",
        "warmup_ratio": config["warmup_ratio"],
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "max_grad_norm": config["max_grad_norm"],
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_target_names": list(config["lora_target_names"]),
        "trainable_parameters": trainable_parameters,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "config_path": str(args.config.resolve()),
        "model_cache_dir": str(model_cache_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume_path": str(resume_path) if resume_path is not None else None,
    }

    save_json(save_dir / "config.json", run_config)

    metrics = {
        "status": "running",
        "initial_validation": {
            "loss": initial_val_loss,
            "perplexity": initial_perplexity,
        },
        "epochs": [],
    }

    save_json(save_dir / "metrics.json", metrics)

    probe_before = probe_parameter.detach().clone()

    for epoch in range(start_epoch, config["num_epochs"]):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        train_start = time.perf_counter()

        avg_loss, epoch_updates = train_one_epoch(
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

        train_seconds = time.perf_counter() - train_start

        peak_allocated_gib = None
        peak_reserved_gib = None

        if device.type == "cuda":
            peak_allocated_gib = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            peak_reserved_gib = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )

        global_step += epoch_updates

        val_loss = evaluation_loss(
            model=model,
            dataloader=val_dataloader,
            device=device,
            use_amp=True,
        )

        valid_perplexity = compute_perplexity(val_loss)

        checkpoint_path = save_dir / f"epoch_{epoch + 1}.pt"

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch+1,
            global_step=global_step,
        )
        print("checkpoint saved:", checkpoint_path)

        adapter_path = (
                save_dir / f"adapter_epoch_{epoch + 1}.pt"
        )

        save_lora_adapter(
            path=adapter_path,
            model=model,
            base_model_name=config["model_name"],
            rank=config["lora_rank"],
            alpha=config["lora_alpha"],
            target_names=config["lora_target_names"],
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "validation_loss": val_loss,
            "validation_perplexity": valid_perplexity,
            "optimizer_updates": epoch_updates,
            "global_step": global_step,
            "train_seconds": train_seconds,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "checkpoint_path": str(checkpoint_path),
            "adapter_path": str(adapter_path),
        }

        metrics["epochs"].append(epoch_metrics)
        save_json(save_dir / "metrics.json", metrics)

        print(f"training time: {train_seconds:.2f} seconds")
        print("peak allocated GPU memory:", peak_allocated_gib, "GiB")

        print("adapter saved:", adapter_path)

        print(
            f"epoch {epoch + 1}/{config["num_epochs"]}, "
            f"train loss: {avg_loss}, "
            f"validation loss: {val_loss}, "
            f"valid perplexity: {valid_perplexity}, "
            f"global step: {global_step}"
        )

    metrics["status"] = "completed"
    save_json(save_dir / "metrics.json", metrics)

    print(
        "final learning rate:",
        scheduler.get_last_lr()[0],
    )

    probe_after = probe_parameter.detach()

    print(
        "probe changed:",
        not torch.equal(probe_before, probe_after),
    )
    print(
        "max probe change:",
        (probe_after - probe_before).abs().max().item(),
    )

if __name__ == "__main__":
    main()
