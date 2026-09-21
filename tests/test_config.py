from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.utils import load_json_config, validate_train_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_config():
    return load_json_config(
        PROJECT_ROOT / "configs" / "lora_r8.json"
    )


def test_valid_config_is_not_modified(valid_config):
    before = deepcopy(valid_config)

    validate_train_config(valid_config)

    assert valid_config == before


@pytest.mark.parametrize(
    "field",
    [
        "model_name", "model_cache_dir", "data_dir", "checkpoint_dir",
        "resume_path", "seed", "num_epochs", "batch_size",
        "gradient_accumulation_steps", "warmup_ratio", "learning_rate",
        "weight_decay", "max_length", "max_grad_norm", "lora_rank",
        "lora_alpha", "lora_target_names",
    ],
)
def test_missing_field_is_rejected(valid_config, field):
    del valid_config[field]

    with pytest.raises(ValueError, match=f"Missing configuration fields:.*{field}"):
        validate_train_config(valid_config)


def test_unknown_field_is_rejected(valid_config):
    valid_config["num_epoch"] = 1

    with pytest.raises(ValueError, match="Unknown configuration fields:.*num_epoch"):
        validate_train_config(valid_config)


@pytest.mark.parametrize(
    "field, value",
    [
        ("gradient_accumulation_steps", 0),
        ("batch_size", True),
        ("num_epochs", 1.5),
        ("lora_rank", -1),
        ("max_length", 1),
        ("seed", 2**32),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("lora_alpha", float("inf")),
        ("max_grad_norm", 0),
        ("weight_decay", -0.01),
        ("warmup_ratio", 1.1),
        ("resume_path", ""),
        ("lora_target_names", []),
        ("lora_target_names", ["q_proj", "q_proj"]),
        ("model_name", "   "),
        ("model_cache_dir", None),
        ("data_dir", 123),
        ("checkpoint_dir", ""),
        ("batch_size", "1"),
        ("seed", -1),
        ("seed", True),
        ("seed", 42.5),
        ("learning_rate", True),
        ("learning_rate", "0.0001"),
        ("lora_alpha", 0),
        ("weight_decay", float("-inf")),
        ("warmup_ratio", -0.1),
        ("resume_path", False),
        ("lora_target_names", "q_proj"),
    ],
)
def test_invalid_values_are_rejected(valid_config, field, value):
    valid_config[field] = value

    with pytest.raises(ValueError, match=field):
        validate_train_config(valid_config)


@pytest.mark.parametrize("warmup_ratio", [0, 1])
@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_valid_boundaries_are_accepted(valid_config, warmup_ratio, seed):
    valid_config["seed"] = seed
    valid_config["weight_decay"] = 0
    valid_config["warmup_ratio"] = warmup_ratio
    valid_config["max_length"] = 2

    validate_train_config(valid_config)


def test_resume_path_is_accepted_without_accessing_checkpoint(valid_config, tmp_path):
    checkpoint = tmp_path / "not_created.pt"
    valid_config["resume_path"] = str(checkpoint)

    validate_train_config(valid_config)

    assert not checkpoint.exists()


@pytest.mark.parametrize("targets", [[""], ["q_proj", "  "], ["q_proj", None]])
def test_invalid_target_names_are_rejected(valid_config, targets):
    valid_config["lora_target_names"] = targets

    with pytest.raises(ValueError, match="LoRA target name"):
        validate_train_config(valid_config)


@pytest.mark.parametrize("config", [None, [], "config"])
def test_validator_rejects_non_objects(config):
    with pytest.raises(ValueError, match="JSON object"):
        validate_train_config(config)


def test_load_json_preserves_values_and_utf8(valid_config, tmp_path):
    valid_config["checkpoint_dir"] = "checkpoint/实验"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config, ensure_ascii=False), encoding="utf-8")

    loaded = load_json_config(path)

    assert loaded == valid_config
    assert loaded["resume_path"] is None
    validate_train_config(loaded)


def test_load_json_accepts_string_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"enabled": true}', encoding="utf-8")

    assert load_json_config(str(path)) == {"enabled": True}


@pytest.mark.parametrize("content", ["null", "[]", "42"])
def test_load_json_rejects_non_objects(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_config(path)


@pytest.mark.parametrize("content", ["", '{"seed": 42,}'])
def test_load_json_rejects_invalid_syntax(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_json_config(path)


def test_load_json_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json_config(tmp_path / "missing.json")
