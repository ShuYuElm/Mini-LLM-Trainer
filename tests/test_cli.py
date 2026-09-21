"""Exercise real argument parsing and entrypoints without loading Qwen."""

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import evaluate
import smoke_train
import train
from src.utils import load_json_config, parse_train_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [train, smoke_train, evaluate]


@pytest.fixture(autouse=True)
def block_expensive_operations(monkeypatch, tmp_path):
    """Fail immediately if an entrypoint unexpectedly reaches real ML work."""
    blocked = []
    for module in ENTRYPOINTS:
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
        for name in ("AutoTokenizer", "AutoModelForCausalLM"):
            loader = Mock(side_effect=AssertionError(f"Unexpected {name} loading"))
            monkeypatch.setattr(module, name, SimpleNamespace(from_pretrained=loader))
            blocked.append(loader)
        for name in (
            "load_from_disk", "set_seed", "train_one_epoch", "evaluation_loss",
            "generate_response", "load_adapter_for_evaluation", "load_checkpoint",
            "save_checkpoint", "save_lora_adapter",
        ):
            if hasattr(module, name):
                operation = Mock(side_effect=AssertionError(f"Unexpected {name}"))
                monkeypatch.setattr(module, name, operation)
                blocked.append(operation)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    yield
    for operation in blocked:
        operation.assert_not_called()


@pytest.fixture
def config_file(tmp_path):
    config = load_json_config(PROJECT_ROOT / "configs" / "lora_r8.json")
    config["checkpoint_dir"] = "checkpoint/new_run"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("module", ENTRYPOINTS, ids=lambda module: module.__name__)
def test_help_exits_before_loading(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__file__, "--help"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 0
    option = "--adapter" if module is evaluate else "--config"
    assert option in capsys.readouterr().out


@pytest.mark.parametrize("module", ENTRYPOINTS, ids=lambda module: module.__name__)
@pytest.mark.parametrize("case", ["required", "missing_file", "directory", "unknown"])
def test_argument_errors_exit_before_loading(module, case, tmp_path, monkeypatch, capsys):
    option = "--adapter" if module is evaluate else "--config"
    existing = tmp_path / "existing.file"
    existing.touch()
    arguments = {
        "required": [],
        "missing_file": [option, str(tmp_path / "missing.file")],
        "directory": [option, str(tmp_path)],
        "unknown": [option, str(existing), "--typo"],
    }[case]
    monkeypatch.setattr(sys, "argv", [module.__file__, *arguments])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    message = capsys.readouterr().err
    if case == "unknown":
        assert "unrecognized arguments: --typo" in message
    elif case == "required":
        assert option in message
        assert "required" in message
    else:
        assert "file does not exist" in message
        assert arguments[1] in message


def test_training_parser_returns_config_path(config_file):
    args = parse_train_args(["--config", str(config_file)])

    assert args.config == config_file
    assert isinstance(args.config, Path)


@pytest.mark.parametrize("module", [train, smoke_train], ids=lambda module: module.__name__)
@pytest.mark.parametrize("case", ["invalid_value", "missing_key", "unknown_key", "malformed_json"])
def test_invalid_config_fails_before_loading(module, case, config_file, monkeypatch):
    config = load_json_config(config_file)
    if case == "invalid_value":
        config["gradient_accumulation_steps"] = 0
        message = "gradient_accumulation_steps"
    elif case == "missing_key":
        del config["num_epochs"]
        message = "Missing configuration fields:.*num_epochs"
    elif case == "unknown_key":
        config["num_epoch"] = 1
        message = "Unknown configuration fields:.*num_epoch"
    else:
        message = "Expecting"
    content = "{" if case == "malformed_json" else json.dumps(config)
    config_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [module.__file__, "--config", str(config_file)])

    with pytest.raises(ValueError, match=message):
        module.main()

    assert not (config_file.parent / "checkpoint").exists()


@pytest.mark.parametrize("path_kind", ["directory", "file"])
def test_fresh_training_preserves_existing_output(path_kind, config_file, tmp_path, monkeypatch):
    output = tmp_path / "checkpoint" / "new_run"
    output.parent.mkdir()
    if path_kind == "directory":
        output.mkdir()
        sentinel = output / "metrics.json"
    else:
        sentinel = output
    sentinel.write_bytes(b"existing experiment")
    # Configured relative paths must remain relative to PROJECT_ROOT, not CWD.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_file)])

    with pytest.raises(FileExistsError, match="choose a new run directory"):
        train.main()

    assert sentinel.read_bytes() == b"existing experiment"
    assert not (elsewhere / "checkpoint").exists()


def test_smoke_rejects_resume_before_loading(config_file, monkeypatch):
    config = load_json_config(config_file)
    config["resume_path"] = "checkpoint/epoch_1.pt"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke_train.py", "--config", str(config_file)])

    with pytest.raises(ValueError, match="resume_path must be null"):
        smoke_train.main()


@pytest.mark.parametrize("answer_only", [False, True], ids=["full_sequence", "answer_only"])
def test_evaluation_cli_routes_policy_and_report(answer_only, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    adapter = run_dir / "adapter.pt"
    adapter.write_bytes(b"adapter placeholder")
    report_name = "evaluation_answer_only.json" if answer_only else "evaluation.json"
    other_name = "evaluation.json" if answer_only else "evaluation_answer_only.json"
    other_report = run_dir / other_name
    other_report.write_bytes(b"preserve the other policy report")
    prompts = [{"id": "example", "instruction": "Say hello.", "input": ""}]
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    monkeypatch.setattr(evaluate, "PROMPT_PATH", prompt_path)
    monkeypatch.setattr(evaluate, "DATA_DIR", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    arguments = ["evaluate.py", "--adapter", "run/adapter.pt"]
    if answer_only:
        arguments.append("--answer-only")
    monkeypatch.setattr(sys, "argv", arguments)

    tokenizer = SimpleNamespace(pad_token_id=0)
    model = SimpleNamespace(to=Mock(), adapted=False)
    monkeypatch.setattr(
        evaluate, "AutoTokenizer",
        SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
    )
    monkeypatch.setattr(
        evaluate, "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    raw_samples = [{} for _ in range(200)]
    monkeypatch.setattr(evaluate, "load_from_disk", Mock(return_value={"validation": raw_samples}))
    dataset_factory = Mock(return_value=raw_samples)
    monkeypatch.setattr(evaluate, "SFTDataset", dataset_factory)
    events = []
    loaders = []
    metadata = {
        "base_model_name": evaluate.MODEL_NAME, "rank": 8, "alpha": 16,
        "target_names": ["q_proj", "v_proj"],
    }

    def fake_loss(**kwargs):
        assert kwargs["model"] is model
        loaders.append(kwargs["dataloader"])
        events.append("lora_loss" if model.adapted else "base_loss")
        return 1.0 if model.adapted else 2.0

    def fake_generate(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["tokenizer"] is tokenizer
        assert kwargs["instruction"] == prompts[0]["instruction"]
        assert kwargs["input_text"] == ""
        assert kwargs["max_input_length"] == evaluate.MAX_INPUT_LENGTH
        assert kwargs["max_new_tokens"] == evaluate.MAX_NEW_TOKENS
        response = "lora response" if model.adapted else "base response"
        events.append(response)
        return response

    def fake_load_adapter(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["adapter_path"] == adapter.resolve()
        assert kwargs["base_model_name"] == evaluate.MODEL_NAME
        events.append("load_adapter")
        model.adapted = True
        return metadata

    monkeypatch.setattr(evaluate, "evaluation_loss", fake_loss)
    monkeypatch.setattr(evaluate, "generate_response", fake_generate)
    monkeypatch.setattr(evaluate, "load_adapter_for_evaluation", fake_load_adapter)

    evaluate.main()

    dataset_factory.assert_called_once_with(
        raw_samples, tokenizer, max_length=evaluate.MAX_LENGTH,
        answer_only=answer_only,
    )
    assert loaders[0] is loaders[1]
    assert events == ["base_loss", "base response", "load_adapter", "lora_loss", "lora response"]
    report = load_json_config(run_dir / report_name)
    assert report["label_policy"] == ("answer_only" if answer_only else "full_sequence")
    assert report["adapter_path"] == str(adapter.resolve())
    assert report["adapter_metadata"] == metadata
    assert report["validation_samples"] == 200
    assert report["prompts"] == prompts
    assert report["generation"] == {
        "max_input_length": evaluate.MAX_INPUT_LENGTH,
        "max_new_tokens": evaluate.MAX_NEW_TOKENS,
        "do_sample": False,
    }
    for key, loss in [("base", 2.0), ("lora", 1.0)]:
        assert report[key]["loss"] == loss
        assert report[key]["perplexity"] == pytest.approx(math.exp(loss))
        assert report[key]["samples"] == [{"id": "example", "response": f"{key} response"}]
    assert other_report.read_bytes() == b"preserve the other policy report"
    assert adapter.read_bytes() == b"adapter placeholder"
