from pathlib import Path

import argparse
import torch
import random
import json
import math
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

def save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scheduler_state = (
        scheduler.state_dict()
        if scheduler is not None
        else None
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "epoch": epoch,
        "global_step": global_step,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler=None):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    scheduler_state = checkpoint["scheduler"]

    if (scheduler is None) != (scheduler_state is None):
        raise ValueError(
            "scheduler configuration does not match checkpoint"
        )

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    return epoch, global_step


def save_lora_adapter(path, model, base_model_name, rank, alpha, target_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    adapter = {}
    for name, parameter in model.named_parameters():
        if name.endswith((".lora_A.weight", ".lora_B.weight")):
            adapter[name] = parameter.clone().detach().cpu()

    if not adapter:
        raise ValueError("model has no LoRA adapter parameters")

    payload = {
        "base_model_name": base_model_name,
        "rank": rank,
        "alpha": alpha,
        "target_names": list(target_names),
        "adapter": adapter,
    }

    torch.save(payload, path)


def load_lora_adapter(path, model):
    path = Path(path)

    payload = torch.load(path, map_location="cpu", weights_only=True)

    saved = payload["adapter"]

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith((".lora_A.weight", ".lora_B.weight"))
    }

    if set(saved) != set(current):
        raise ValueError("adapter configuration does not match model")

    for name in current:
        if current[name].shape != saved[name].shape:
            raise ValueError("adapter configuration does not match model")

    with torch.no_grad():
        for name, parameter in current.items():
            source = saved[name].to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(source)

    return {
        "base_model_name": payload["base_model_name"],
        "rank": payload["rank"],
        "alpha": payload["alpha"],
        "target_names": payload["target_names"],
    }

def load_json_config(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")

    return config

def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Load configuration for training or a smoke check."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training JSON configuration.",
    )

    args = parser.parse_args(argv)

    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")

    return args


def validate_train_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")

    required_keys = {
        "model_name",
        "model_cache_dir",
        "data_dir",
        "checkpoint_dir",
        "resume_path",
        "seed",
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "learning_rate",
        "weight_decay",
        "max_length",
        "max_grad_norm",
        "lora_rank",
        "lora_alpha",
        "lora_target_names",
    }

    missing = required_keys - config.keys()
    unknown = config.keys() - required_keys

    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")

    # 路径和模型名称目前都以 JSON 字符串表示。
    string_fields = (
        "model_name",
        "model_cache_dir",
        "data_dir",
        "checkpoint_dir",
    )

    for name in string_fields:
        value = config[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    resume_path = config["resume_path"]
    if resume_path is not None:
        if not isinstance(resume_path, str) or not resume_path.strip():
            raise ValueError("resume_path must be null or a non-empty string")

    positive_integer_fields = (
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "lora_rank",
    )

    for name in positive_integer_fields:
        value = config[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if config["max_length"] < 2:
        raise ValueError("max_length must be at least 2 for next-token prediction")

    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    numeric_fields = (
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "lora_alpha",
    )

    for name in numeric_fields:
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")

    for name in ("learning_rate", "max_grad_norm", "lora_alpha"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be greater than 0")

    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")

    if not 0 <= config["warmup_ratio"] <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")

    targets = config["lora_target_names"]

    if not isinstance(targets, list) or not targets:
        raise ValueError("lora_target_names must be a non-empty list")

    if any(
        not isinstance(name, str) or not name.strip()
        for name in targets
    ):
        raise ValueError("Each LoRA target name must be a non-empty string")

    if len(targets) != len(set(targets)):
        raise ValueError("lora_target_names must not contain duplicates")
