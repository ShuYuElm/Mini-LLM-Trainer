"""Synthetic reports exercise export validation without local checkpoints."""

import csv
import hashlib
import json
import math

import pytest

from summarize_results import RANKS, export_results, main, write_json


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for rank in RANKS:
        name = f"lora_r{rank}_seed42_baseline_v1"
        config = {
            "model_name": "Qwen/Qwen3-0.6B-Base", "seed": 42,
            "data_dir": "D:/private/project/data/fixed_split", "max_length": 512,
            "train_samples": 2000, "validation_samples": 200, "num_epochs": 1,
            "label_policy": "full_sequence", "lora_rank": rank, "lora_alpha": rank * 2,
            "lora_target_names": ["q_proj", "v_proj"],
            "trainable_parameters": rank * 143360, "num_training_steps": 250,
            "learning_rate": 1e-4,
        }
        adapter_path = f"D:/private/project/checkpoint/{name}/adapter_epoch_1.pt"
        full_loss = 1.3 + 1 / rank
        epoch = {
            "epoch": 1, "global_step": 250, "optimizer_updates": 250,
            "validation_loss": full_loss, "validation_perplexity": math.exp(full_loss),
            "train_loss": 1.5, "train_seconds": 100 + rank,
            "peak_allocated_gib": 3.5, "peak_reserved_gib": 4.5,
            "checkpoint_path": f"D:/private/project/checkpoint/{name}/epoch_1.pt",
            "adapter_path": adapter_path,
        }
        metrics = {"status": "completed", "initial_validation": {"loss": 2.0, "perplexity": math.exp(2.0)}, "epochs": [epoch]}
        write_json(root / name / "config.json", config)
        write_json(root / name / "metrics.json", metrics)
        for policy in ("full_sequence", "answer_only"):
            base_loss = 2.0 if policy == "full_sequence" else 1.6
            loss = full_loss if policy == "full_sequence" else full_loss - 0.1
            report = {
                "model_name": config["model_name"], "data_dir": config["data_dir"],
                "validation_samples": 200, "validation_max_length": 512,
                "label_policy": policy, "adapter_path": adapter_path,
                "adapter_metadata": {"base_model_name": config["model_name"], "rank": rank, "alpha": rank * 2, "target_names": ["q_proj", "v_proj"]},
                "generation": {"max_new_tokens": 128, "do_sample": False},
                "prompts": [{"id": str(i), "instruction": "你好"} for i in range(6)],
            }
            for model, value in (("base", base_loss), ("lora", loss)):
                report[model] = {
                    "loss": value, "perplexity": math.exp(value),
                    "samples": [{"id": str(i), "response": f"你好 {model}: D:/text/is/not/a/path/field"} for i in range(6)],
                }
            filename = "evaluation.json" if policy == "full_sequence" else "evaluation_answer_only.json"
            write_json(root / name / filename, report)
    return root


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_export_preserves_evidence_values_and_sources(source, tmp_path):
    before = snapshot(source)
    output = tmp_path / "export"
    rows = export_results(source, output, plots=False)

    assert snapshot(source) == before
    assert [row["rank"] for row in rows] == [0, 4, 8, 16, 32]
    assert rows[1]["full_sequence_loss"] == 1.55
    assert rows[-1]["trainable_parameters"] == 4587520
    with (output / "rank_comparison.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 5
    assert csv_rows[0]["train_seconds"] == ""
    assert csv_rows[0]["peak_allocated_gib"] == ""
    assert csv_rows[0]["trainable_parameters"] == "0"
    assert float(csv_rows[1]["answer_only_loss"]) == pytest.approx(1.45)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["source_files_relative_to_source_dir"]) == 16
    for record in manifest["source_files_relative_to_source_dir"]:
        relative = record["source"]
        assert record["sha256"] == hashlib.sha256(before[relative]).hexdigest()
        exported = output / "runs" / relative
        assert record["exported_sha256"] == hashlib.sha256(exported.read_bytes()).hexdigest()
        payload = json.loads(exported.read_text(encoding="utf-8"))
        original = json.loads(before[relative])
        if relative.endswith("evaluation.json"):
            assert payload["prompts"] == original["prompts"]
            assert payload["base"] == original["base"]
            assert payload["lora"] == original["lora"]
            assert payload["adapter_path"].startswith("checkpoint/")
            assert payload["data_dir"] == "data/fixed_split"
    first = snapshot(output)
    export_results(source, output, plots=False)
    assert snapshot(output) == first
    # A recipient can re-export from portable reports without checkpoint/.
    assert export_results(output / "runs", tmp_path / "reexport", plots=False) == rows


@pytest.mark.parametrize(
    "filename, keys, value, message",
    [
        ("metrics.json", ["status"], "running", "incomplete"),
        ("config.json", ["seed"], 43, "seed"),
        ("config.json", ["learning_rate"], 0.1, "controls"),
        ("metrics.json", ["epochs", 0, "global_step"], 249, "update count"),
        ("metrics.json", ["epochs", 0, "train_seconds"], -1, "train_seconds"),
        ("evaluation.json", ["lora", "perplexity"], 123, "perplexity"),
        ("evaluation.json", ["lora", "loss"], float("nan"), "nonfinite"),
        ("metrics.json", ["epochs", 0, "train_seconds"], float("inf"), "train_seconds"),
        ("evaluation.json", ["adapter_metadata", "alpha"], 999, "metadata"),
        ("evaluation.json", ["adapter_path"], "X:/different/adapter.pt", "adapter path"),
        ("evaluation_answer_only.json", ["label_policy"], "full_sequence", "label policy"),
        ("evaluation_answer_only.json", ["lora", "samples", 0, "response"], "changed", "responses differ"),
        ("evaluation_answer_only.json", ["lora", "samples", 0, "id"], "bad", "sample IDs"),
    ],
)
def test_inconsistent_report_rejected_before_export(source, tmp_path, filename, keys, value, message):
    path = source / "lora_r8_seed42_baseline_v1" / filename
    report = json.loads(path.read_text(encoding="utf-8"))
    target = report
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    # Deliberately allow invalid JSON numeric extensions to test rejection.
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        export_results(source, tmp_path / "output", plots=False)

    assert not (tmp_path / "output").exists()


def test_missing_report_leaves_existing_export_untouched(source, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "rank_comparison.csv").write_bytes(b"previous export")
    (source / "lora_r32_seed42_baseline_v1" / "evaluation_answer_only.json").unlink()

    with pytest.raises(FileNotFoundError):
        export_results(source, output, plots=False)

    assert snapshot(output) == {"rank_comparison.csv": b"previous export"}


@pytest.mark.parametrize("location", ["same", "child", "parent"])
def test_source_output_overlap_rejected(source, location):
    output = {"same": source, "child": source / "export", "parent": source.parent}[location]
    before = snapshot(source)
    with pytest.raises(ValueError, match="must not overlap"):
        export_results(source, output, plots=False)
    assert snapshot(source) == before


def test_cli_creates_standalone_plots(source, tmp_path, capsys):
    output = tmp_path / "plots"
    main(["--source-dir", str(source), "--output-dir", str(output)])
    assert "Exported 5 comparison rows" in capsys.readouterr().out
    for name in ("validation_loss", "training_resources"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in (output / f"{name}.svg").read_text(encoding="utf-8")
