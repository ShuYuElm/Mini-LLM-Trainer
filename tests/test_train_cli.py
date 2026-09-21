"""CLI wiring tests with tiny CPU modules and stubbed epoch execution.

Real LoRA injection, AdamW construction, and temporary artifact serialization
are exercised. These tests do not run the Qwen training experiment.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import smoke_train
import train
from src.utils import load_json_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyQwenStructure(nn.Module):
    def __init__(self):
        super().__init__()
        attention = nn.Module()
        attention.q_proj = nn.Linear(4, 4, bias=False)
        attention.v_proj = nn.Linear(4, 2, bias=False)
        block = nn.Module()
        block.self_attn = attention
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([block])


class LengthDataset:
    def __init__(self, samples, tokenizer, max_length, answer_only=False):
        self.samples = samples
        self.max_length = max_length
        self.answer_only = answer_only

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.samples[index]  # Preserve the Dataset indexing contract.
        ids = torch.ones(self.max_length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.clone(), "labels": ids.clone()}


@pytest.fixture
def cli_environment(tmp_path, monkeypatch):
    config = load_json_config(PROJECT_ROOT / "configs" / "lora_r8.json")
    config.update(model_cache_dir="cache", data_dir="data", checkpoint_dir="runs/test")
    splits = {
        "train": [{"sample_id": i} for i in range(2000)],
        "validation": [{"sample_id": i + 2000} for i in range(200)],
    }
    tokenizer = SimpleNamespace(pad_token_id=0)
    model = TinyQwenStructure().to(dtype=torch.bfloat16)
    model_loader = Mock(return_value=model)
    tokenizer_loader = Mock(return_value=tokenizer)
    data_loader = Mock(return_value=splits)
    scheduler = Mock()
    scheduler.get_last_lr.return_value = [0.0]
    scheduler.state_dict.return_value = {"test_scheduler": True}
    scheduler_factory = Mock(return_value=scheduler)
    dataset_factory = Mock(side_effect=LengthDataset)
    seed = Mock()
    for module in (train, smoke_train):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=tokenizer_loader))
        monkeypatch.setattr(module, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=model_loader))
        monkeypatch.setattr(module, "load_from_disk", data_loader)
        monkeypatch.setattr(module, "SFTDataset", dataset_factory)
        monkeypatch.setattr(module, "get_linear_schedule_with_warmup", scheduler_factory)
        monkeypatch.setattr(module, "set_seed", seed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return SimpleNamespace(
        root=tmp_path, config=config, model=model, seed=seed, splits=splits,
        model_loader=model_loader, tokenizer_loader=tokenizer_loader,
        data_loader=data_loader, scheduler=scheduler,
        scheduler_factory=scheduler_factory, dataset_factory=dataset_factory,
    )


def invoke(module, environment, monkeypatch):
    config_path = environment.root / "config.json"
    config_path.write_text(json.dumps(environment.config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [module.__file__, "--config", str(config_path)])
    module.main()
    return config_path


def assert_epoch_arguments(kwargs, environment):
    config = environment.config
    assert kwargs["model"] is environment.model
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["use_amp"] is True
    assert kwargs["gradient_accumulation_steps"] == config["gradient_accumulation_steps"]
    assert kwargs["max_grad_norm"] == config["max_grad_norm"]
    assert kwargs["scheduler"] is environment.scheduler
    optimizer = kwargs["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == config["learning_rate"]
    assert optimizer.defaults["weight_decay"] == config["weight_decay"]
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = {id(p) for p in environment.model.parameters() if p.requires_grad}
    assert actual == expected


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_training_config_reaches_epoch_loop_and_saved_artifacts(resume, cli_environment, monkeypatch):
    env = cli_environment
    env.config.update(
        seed=7, batch_size=2, gradient_accumulation_steps=4, num_epochs=2,
        max_length=96, learning_rate=0.0003, weight_decay=0.02,
        max_grad_norm=0.7, lora_rank=4, lora_alpha=12, warmup_ratio=0.2,
    )
    if resume:
        env.config["resume_path"] = "prior/epoch_1.pt"
    calls = []
    events = []
    base_weights = {name: p.detach().clone() for name, p in env.model.named_parameters()}

    def fake_resume(**kwargs):
        assert kwargs["path"] == env.root / "prior" / "epoch_1.pt"
        assert kwargs["model"] is env.model
        assert kwargs["scheduler"] is env.scheduler
        events.append("resume")
        return 1, 250

    resume_loader = Mock(side_effect=fake_resume)
    monkeypatch.setattr(train, "load_checkpoint", resume_loader)

    def fake_validation(**kwargs):
        assert kwargs["model"] is env.model
        assert len(kwargs["dataloader"].dataset) == 200
        assert kwargs["dataloader"].batch_size == 1
        events.append("validation")
        return 1.25

    def fake_epoch(**kwargs):
        assert_epoch_arguments(kwargs, env)
        loader = kwargs["dataloader"]
        assert loader.batch_size == 2 and len(loader) == 1000
        assert not loader.dataset.answer_only
        assert list(loader.sampler) == list(range(2000))
        calls.append(kwargs)
        events.append("epoch")
        # Simulate an adapter update; optimizer mechanics have separate tests.
        with torch.no_grad():
            env.model.model.layers[0].self_attn.q_proj.lora_B.weight.add_(0.125)
        return 1.5, 250

    monkeypatch.setattr(train, "evaluation_loss", fake_validation)
    monkeypatch.setattr(train, "train_one_epoch", fake_epoch)
    config_path = invoke(train, env, monkeypatch)

    env.seed.assert_called_once_with(7)
    env.tokenizer_loader.assert_called_once_with(env.config["model_name"], cache_dir=env.root / "cache")
    env.model_loader.assert_called_once_with(
        env.config["model_name"], cache_dir=env.root / "cache", dtype=torch.bfloat16,
    )
    env.data_loader.assert_called_once_with(str(env.root / "data"))
    assert len(env.dataset_factory.call_args_list) == 2
    assert all(call.kwargs["max_length"] == 96 for call in env.dataset_factory.call_args_list)
    env.scheduler_factory.assert_called_once_with(
        optimizer=calls[0]["optimizer"], num_warmup_steps=100, num_training_steps=500,
    )
    if resume:
        resume_loader.assert_called_once()
        assert resume_loader.call_args.kwargs["optimizer"] is calls[0]["optimizer"]
    else:
        resume_loader.assert_not_called()
    assert events == (["resume"] if resume else []) + ["validation"] + ["epoch", "validation"] * (1 if resume else 2)

    output = env.root / "runs" / "test"
    if resume:
        output /= "resumed"
    effective = load_json_config(output / "config.json")
    for key in (
        "seed", "batch_size", "gradient_accumulation_steps", "num_epochs",
        "max_length", "learning_rate", "weight_decay", "max_grad_norm",
        "lora_rank", "lora_alpha", "lora_target_names", "warmup_ratio",
    ):
        assert effective[key] == env.config[key]
    assert effective["config_path"] == str(config_path)
    assert effective["data_dir"] == str(env.root / "data")
    assert effective["model_cache_dir"] == str(env.root / "cache")
    assert effective["checkpoint_dir"] == str(env.root / "runs" / "test")
    expected_resume = str(env.root / "prior" / "epoch_1.pt") if resume else None
    assert effective["resume_path"] == expected_resume
    assert effective["num_training_steps"] == 500
    assert effective["num_warmup_steps"] == 100
    assert effective["trainable_parameters"] == 4 * ((4 + 4) + (4 + 2))
    metrics = load_json_config(output / "metrics.json")
    assert metrics["status"] == "completed"
    expected_epochs = [2] if resume else [1, 2]
    assert [row["epoch"] for row in metrics["epochs"]] == expected_epochs
    assert [row["global_step"] for row in metrics["epochs"]] == ([500] if resume else [250, 500])
    for row in metrics["epochs"]:
        assert row["optimizer_updates"] == 250
        assert row["train_loss"] == 1.5 and row["validation_loss"] == 1.25
        assert row["peak_allocated_gib"] is None
        checkpoint_path = output / f"epoch_{row['epoch']}.pt"
        adapter_path = output / f"adapter_epoch_{row['epoch']}.pt"
        assert row["checkpoint_path"] == str(checkpoint_path)
        assert row["adapter_path"] == str(adapter_path)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        adapter = torch.load(adapter_path, weights_only=True)
        assert checkpoint["epoch"] == row["epoch"]
        assert checkpoint["global_step"] == row["global_step"]
        assert adapter["rank"] == 4 and adapter["alpha"] == 12
        assert adapter["target_names"] == ["q_proj", "v_proj"]
        for name, weight in adapter["adapter"].items():
            assert torch.equal(weight, checkpoint["model"][name])
    for name, weight in base_weights.items():
        wrapped_name = name.replace(".weight", ".base_layer.weight")
        assert torch.equal(env.model.state_dict()[wrapped_name], weight)
    assert env.model.model.layers[0].self_attn.q_proj.scaling == 3
    assert not (env.root / "elsewhere" / "runs").exists()


@pytest.mark.parametrize("rank", [4, 16])
def test_smoke_cli_uses_config_and_full_schedule_without_artifacts(rank, cli_environment, monkeypatch, capsys):
    env = cli_environment
    env.config.update(lora_rank=rank, lora_alpha=rank * 2, max_grad_norm=0.7)
    calls = []

    def fake_epoch(**kwargs):
        assert_epoch_arguments(kwargs, env)
        loader = kwargs["dataloader"]
        assert len(loader) == 16 and loader.batch_size == 1
        assert loader.dataset.indices == list(range(16))
        assert loader.dataset.dataset.max_length == 512
        calls.append(kwargs)
        with torch.no_grad():
            env.model.model.layers[0].self_attn.q_proj.lora_B.weight.add_(0.125)
        return 1.5, 2

    monkeypatch.setattr(smoke_train, "train_one_epoch", fake_epoch)
    invoke(smoke_train, env, monkeypatch)

    assert len(calls) == 1
    env.seed.assert_called_once_with(env.config["seed"])
    env.tokenizer_loader.assert_called_once_with(
        env.config["model_name"], cache_dir=env.root / "cache", local_files_only=True,
    )
    env.model_loader.assert_called_once_with(
        env.config["model_name"], cache_dir=env.root / "cache",
        local_files_only=True, dtype=torch.bfloat16,
    )
    env.data_loader.assert_called_once_with(str(env.root / "data"))
    env.scheduler_factory.assert_called_once_with(
        calls[0]["optimizer"], num_warmup_steps=25, num_training_steps=250,
    )
    assert sum(p.numel() for p in env.model.parameters() if p.requires_grad) == rank * 14
    assert "Smoke training passed. No checkpoint saved." in capsys.readouterr().out
    assert not (env.root / "runs").exists()
    assert list(env.root.rglob("*.pt")) == []
