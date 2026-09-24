"""Quality generation entry point: prompt isolation, review export, and format checks."""

import json
from pathlib import Path

import pytest

import generate_quality_eval as quality
from audit_quality_eval import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "configs/quality_eval_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def prompt_file(tmp_path, payload):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def make_adapter(directory, run_name):
    run_dir = Path(directory) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "adapter_epoch_1.pt"
    path.write_bytes(b"stub adapter")
    return path


def stub_generation(monkeypatch, responder=None, seen=None):
    """Replace model loading and generation; keep the caller's prompt payload."""
    respond = responder if responder is not None else (lambda prompt_id: f"answer::{prompt_id}")

    def fake_model(adapter_path, device):
        metadata = None if adapter_path is None else {"rank": 8, "alpha": 16, "target_names": ["q_proj", "v_proj"]}
        return object(), metadata

    def fake_generate(model, tokenizer, prompts, plan, device):
        if seen is not None:
            seen.append({"prompts": [dict(prompt) for prompt in prompts], "plan": dict(plan)})
        return [{"id": prompt["id"], "response": respond(prompt["id"])} for prompt in prompts]

    monkeypatch.setattr(quality, "load_tokenizer", lambda: object())
    monkeypatch.setattr(quality, "load_run_model", fake_model)
    monkeypatch.setattr(quality, "generate_run_responses", fake_generate)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_workbook_lists_every_candidate_and_hides_identities(tmp_path, payload, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    adapter = make_adapter(tmp_path, "lora_r8_seed42_baseline_v1")
    output = tmp_path / "out"

    quality.main(["--prompts", str(prompt_file), "--adapter", str(adapter), "--output-dir", str(output)])

    text = (output / quality.WORKBOOK).read_text(encoding="utf-8")

    assert f"<!-- prompt_set_sha256: {canonical_hash(payload)} -->" in text
    assert text.count("### 候选 ") == 40 * 2
    assert text.count(": _ _ _ _ |") == 40 * 2
    assert text.count("answer::") == 40 * 2
    assert "lora_r8_seed42_baseline_v1" not in text
    assert '"base"' not in text


def test_workbook_keeps_multiline_responses_inside_one_fence(payload):
    report = sample_report(payload, ["base"], count=1)
    report["runs"][0]["samples"][0]["response"] = "- intake\n- sorting"

    text = quality.build_review_workbook(payload, report, 11)

    assert "```text\n- intake\n- sorting\n```" in text


def test_workbook_fence_survives_a_response_containing_backticks(payload):
    report = sample_report(payload, ["base"], count=1)
    report["runs"][0]["samples"][0]["response"] = "```json\n{}\n```"

    text = quality.build_review_workbook(payload, report, 11)

    assert "````text\n```json\n{}\n```\n````" in text


def test_generation_writes_report_and_review_files(tmp_path, payload, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    adapter = make_adapter(tmp_path, "lora_r8_seed42_baseline_v1")
    output = tmp_path / "out"

    written = quality.main([
        "--prompts", str(prompt_file),
        "--adapter", str(adapter),
        "--output-dir", str(output),
    ])

    assert [Path(path).name for path in written] == [
        quality.GENERATION_REPORT, *quality.REVIEW_FILES,
    ]

    report = read_json(output / quality.GENERATION_REPORT)
    assert report["evaluation_id"] == "quality_eval_v1"
    assert report["prompt_set_sha256"] == canonical_hash(payload)
    assert report["prompt_count"] == 40
    assert report["complete_prompt_set"] is True
    assert report["generation"] == {
        "max_input_length": 256, "max_new_tokens": 128, "do_sample": False,
        "device": report["generation"]["device"], "dtype": "bfloat16", "use_amp": True,
    }
    assert [record["run_id"] for record in report["runs"]] == ["base", "lora_r8_seed42_baseline_v1"]
    assert report["runs"][0]["adapter_path"] is None
    assert report["runs"][0]["adapter_metadata"] is None
    assert report["runs"][1]["adapter_metadata"]["rank"] == 8
    assert [sample["id"] for sample in report["runs"][0]["samples"]] == report["prompt_ids"]

    blind = read_json(output / quality.BLIND_REVIEW)
    mapping = read_json(output / quality.REVIEW_MAPPING)
    sheet = read_json(output / quality.SCORE_SHEET)

    assert len(blind["answers"]) == 40
    assert blind["answers"][0]["prompt_id"] == report["prompt_ids"][0]
    assert blind["answers"][0]["reference_answer"] == payload["prompts"][0]["reference_answer"]
    assert len(sheet["entries"]) == 40 * 2

    responses = {
        record["run_id"]: {sample["id"]: sample["response"] for sample in record["samples"]}
        for record in report["runs"]
    }

    for answer in blind["answers"]:
        assignment = mapping["labels"][answer["prompt_id"]]
        assert [candidate["label"] for candidate in answer["candidates"]] == ["A", "B"]
        assert sorted(assignment.values()) == ["base", "lora_r8_seed42_baseline_v1"]
        for candidate in answer["candidates"]:
            run_id = assignment[candidate["label"]]
            assert candidate["response"] == responses[run_id][answer["prompt_id"]]


def test_blind_file_hides_model_identities(tmp_path, payload, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    adapter = make_adapter(tmp_path, "lora_r8_seed42_baseline_v1")
    output = tmp_path / "out"

    quality.main(["--prompts", str(prompt_file), "--adapter", str(adapter), "--output-dir", str(output)])

    blind_text = (output / quality.BLIND_REVIEW).read_text(encoding="utf-8")
    mapping = read_json(output / quality.REVIEW_MAPPING)

    assert "lora_r8_seed42_baseline_v1" not in blind_text
    assert '"base"' not in blind_text
    assert "lora_r8_seed42_baseline_v1" in json.dumps(mapping)


def test_generation_runs_once_per_run_with_the_configured_prompt_set(tmp_path, payload, prompt_file, monkeypatch):
    seen = []
    stub_generation(monkeypatch, seen=seen)
    adapter = make_adapter(tmp_path, "run_a")

    quality.main(["--prompts", str(prompt_file), "--adapter", str(adapter), "--output-dir", str(tmp_path / "out")])

    assert len(seen) == 2

    for call in seen:
        assert [prompt["id"] for prompt in call["prompts"]] == [prompt["id"] for prompt in payload["prompts"]]
        assert call["plan"] == payload["generation_plan"]


def test_generate_response_receives_only_instruction_and_input(payload, monkeypatch):
    calls = []

    def fake_generate_response(**kwargs):
        calls.append(kwargs)
        return "generated text"

    monkeypatch.setattr(quality, "generate_response", fake_generate_response)

    prompts = payload["prompts"]
    samples = quality.generate_run_responses(object(), object(), prompts, payload["generation_plan"], "cpu")

    assert [sample["id"] for sample in samples] == [prompt["id"] for prompt in prompts]
    assert all(sample["response"] == "generated text" for sample in samples)
    assert len(calls) == len(prompts)

    for call, prompt in zip(calls, prompts):
        assert set(call) == {
            "model", "tokenizer", "instruction", "input_text", "device",
            "max_input_length", "max_new_tokens", "use_amp",
        }
        assert call["instruction"] == prompt["instruction"]
        assert call["input_text"] == prompt["input"]
        assert call["max_input_length"] == 256
        assert call["max_new_tokens"] == 128

        # Criteria are never handed to the model, as a field or as a value.
        assert prompt["reference_answer"] not in call["instruction"] + "\n" + call["input_text"]
        for value in call.values():
            assert value != prompt["required_facts"]
            assert value != prompt["prohibited_additions"]


def test_run_specs_include_base_first_and_name_runs_by_directory(tmp_path):
    first = make_adapter(tmp_path, "lora_r8_seed42_baseline_v1")
    second = make_adapter(tmp_path, "lora_r32_seed42_baseline_v1")

    specs = quality.run_specs([first, second])

    assert [spec["run_id"] for spec in specs] == [
        "base", "lora_r8_seed42_baseline_v1", "lora_r32_seed42_baseline_v1",
    ]
    assert specs[0]["adapter"] is None
    assert specs[1]["adapter"] == first


def test_duplicate_run_ids_are_rejected(tmp_path):
    left = make_adapter(tmp_path / "one", "same_name")
    right = make_adapter(tmp_path / "two", "same_name")

    with pytest.raises(ValueError, match="duplicate run ids"):
        quality.run_specs([left, right])


def test_existing_outputs_are_preserved_unless_overwritten(tmp_path, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    existing = output / quality.GENERATION_REPORT
    existing.write_text('{"kept": true}', encoding="utf-8")

    argv = ["--prompts", str(prompt_file), "--output-dir", str(output)]

    with pytest.raises(FileExistsError, match="existing output files"):
        quality.main(argv)

    assert read_json(existing) == {"kept": True}

    quality.main(argv + ["--overwrite"])

    assert read_json(existing)["prompt_count"] == 40


def test_limit_records_a_partial_report(tmp_path, payload, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    output = tmp_path / "out"

    quality.main(["--prompts", str(prompt_file), "--output-dir", str(output), "--limit", "3"])

    report = read_json(output / quality.GENERATION_REPORT)
    blind = read_json(output / quality.BLIND_REVIEW)
    sheet = read_json(output / quality.SCORE_SHEET)

    assert report["prompt_count"] == 3
    assert report["prompt_set_prompt_count"] == 40
    assert report["complete_prompt_set"] is False
    assert report["prompt_ids"] == [prompt["id"] for prompt in payload["prompts"][:3]]
    assert blind["complete_prompt_set"] is False
    assert len(blind["answers"]) == 3
    assert [candidate["label"] for candidate in blind["answers"][0]["candidates"]] == ["A"]
    assert len(sheet["entries"]) == 3


def test_from_report_reexports_without_loading_a_model(tmp_path, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    adapter = make_adapter(tmp_path, "run_a")
    first = tmp_path / "first"

    quality.main(["--prompts", str(prompt_file), "--adapter", str(adapter), "--output-dir", str(first)])

    report_path = first / quality.GENERATION_REPORT
    before = report_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("model loading is not allowed while re-exporting")

    monkeypatch.setattr(quality, "load_run_model", forbidden)
    monkeypatch.setattr(quality, "load_tokenizer", forbidden)

    second = tmp_path / "second"

    quality.main(["--prompts", str(prompt_file), "--from-report", str(report_path), "--output-dir", str(second)])

    assert not (second / quality.GENERATION_REPORT).exists()

    for name in quality.REVIEW_FILES:
        assert (second / name).is_file()

    assert report_path.read_bytes() == before
    assert (second / quality.BLIND_REVIEW).read_bytes() == (first / quality.BLIND_REVIEW).read_bytes()


def test_from_report_rejects_a_mismatched_prompt_hash(tmp_path, prompt_file, monkeypatch):
    stub_generation(monkeypatch)
    output = tmp_path / "out"

    quality.main(["--prompts", str(prompt_file), "--output-dir", str(output)])

    report_path = output / quality.GENERATION_REPORT
    report = read_json(report_path)
    report["prompt_set_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt-set hash"):
        quality.main([
            "--prompts", str(prompt_file),
            "--from-report", str(report_path),
            "--output-dir", str(tmp_path / "again"),
        ])


def test_missing_responses_in_a_report_are_rejected(payload):
    report = {
        "prompt_ids": ["sum_workshop", "sum_deliveries"],
        "runs": [{"run_id": "base", "samples": [{"id": "sum_workshop", "response": "text"}]}],
    }

    with pytest.raises(ValueError, match="has no response for 1 prompts"):
        quality.indexed_responses(report, report["prompt_ids"])


def sample_report(payload, run_ids, count=5):
    prompts = payload["prompts"][:count]
    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_count": count,
        "complete_prompt_set": False,
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "runs": [
            {
                "run_id": run_id,
                "samples": [{"id": prompt["id"], "response": f"{run_id}::{prompt['id']}"} for prompt in prompts],
            }
            for run_id in run_ids
        ],
    }


def test_blind_order_is_seeded_and_mapping_matches_candidates(payload):
    run_ids = ["base", "lora_r8_seed42_baseline_v1", "lora_r32_seed42_baseline_v1"]
    report = sample_report(payload, run_ids)

    first = quality.build_blind_review(payload, report, 11)
    again = quality.build_blind_review(payload, report, 11)
    mapping = quality.build_review_mapping(payload, report, 11)

    assert first == again

    for answer in first["answers"]:
        assignment = mapping["labels"][answer["prompt_id"]]
        assert sorted(assignment) == ["A", "B", "C"]
        assert sorted(assignment.values()) == sorted(run_ids)
        assert [candidate["label"] for candidate in answer["candidates"]] == ["A", "B", "C"]
        for candidate in answer["candidates"]:
            assert candidate["response"] == f"{assignment[candidate['label']]}::{answer['prompt_id']}"

    assert quality.blind_assignments(report["prompt_ids"], run_ids, 11) != \
        quality.blind_assignments(report["prompt_ids"], run_ids, 12)


def test_score_sheet_lists_every_dimension_per_candidate(payload):
    report = sample_report(payload, ["base", "lora_r8_seed42_baseline_v1"])

    sheet = quality.build_score_sheet(payload, report)

    assert len(sheet["entries"]) == 10
    assert sheet["entries"][0]["prompt_id"] == report["prompt_ids"][0]
    assert [entry["label"] for entry in sheet["entries"][:2]] == ["A", "B"]
    for entry in sheet["entries"]:
        assert list(entry["scores"]) == list(payload["rubric"])
        assert set(entry["scores"].values()) == {None}
        assert entry["rationale"] == ""


@pytest.mark.parametrize("response, expected_result", [
    ('{"order_id": "T-582", "customer": "Leena Park", "quantity": 14}', True),
    ('  {"quantity": 14, "customer": "Leena Park", "order_id": "T-582"}  ', True),
    ('```json\n{"order_id": "T-582", "customer": "Leena Park", "quantity": 14}\n```', False),
    ('{"order_id": "T-582", "customer": "Leena Park", "quantity": "14"}', False),
    ('{"order_id": "T-582", "customer": "Leena Park"}', False),
    ('{"order_id": "T-582", "customer": "Leena Park", "quantity": 14, "note": "x"}', False),
    ('[{"order_id": "T-582"}]', False),
    ('', False),
])
def test_json_object_check_requires_one_bare_typed_object(response, expected_result):
    expected = {"order_id": "T-582", "customer": "Leena Park", "quantity": 14}

    passed, reason = quality.check_json_object(response, expected)

    assert passed is expected_result
    assert bool(reason) is not expected_result


@pytest.mark.parametrize("response, expected_result", [
    ("- intake\n- sorting\n- repair\n- dispatch", True),
    ("  - intake\r\n- sorting\r\n- repair\r\n- dispatch\r\n", True),
    ("- intake\n- sorting\n- repair", False),
    ("- intake\n-  sorting\n- repair\n- dispatch", False),
    ("- intake\n- sorting\n- repair\n- dispatch\n- archive", False),
    ("Here are the labels:\n- intake\n- sorting\n- repair\n- dispatch", False),
    ("", False),
])
def test_exact_lines_check_allows_outer_whitespace_only(response, expected_result):
    expected = ["- intake", "- sorting", "- repair", "- dispatch"]

    passed, reason = quality.check_exact_lines(response, expected)

    assert passed is expected_result
    assert bool(reason) is not expected_result


def test_automated_checks_pass_reference_answers_and_report_failures(payload):
    prompts = payload["prompts"]
    references = {prompt["id"]: prompt["reference_answer"] for prompt in prompts}

    def record(responder, run_id):
        return {
            "run_id": run_id,
            "samples": [{"id": prompt["id"], "response": responder(prompt["id"])} for prompt in prompts],
        }

    passed_all = quality.run_automated_checks(prompts, [record(lambda name: references[name], "base")])
    checks = passed_all["checks"]

    assert checks["json_object"]["tasks"] == 8
    assert checks["json_object"]["answers"] == 8
    assert checks["json_object"]["passed"] == 8
    assert checks["exact_lines"]["passed"] == 8
    assert checks["json_object"]["failures"] == []

    def fenced(prompt_id):
        return f"```json\n{references[prompt_id]}\n```" if prompt_id == "extract_order" else references[prompt_id]

    failed_one = quality.run_automated_checks(prompts, [record(fenced, "lora_r8")])
    failures = failed_one["checks"]["json_object"]["failures"]

    assert failed_one["checks"]["json_object"]["passed"] == 7
    assert failures == [{
        "run_id": "lora_r8",
        "prompt_id": "extract_order",
        "reason": "response is not a single JSON value",
    }]


@pytest.mark.parametrize("argv, message", [
    (["--prompts", "missing.json"], "Prompt set does not exist"),
    (["--adapter", "missing.pt"], "Adapter file does not exist"),
    (["--limit", "0"], "--limit must be positive"),
    (["--seed", "-1"], "--seed must be an integer"),
])
def test_cli_rejects_invalid_arguments(tmp_path, prompt_file, argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        quality.main(["--prompts", str(prompt_file), *argv])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_adapters_with_from_report(tmp_path, prompt_file, capsys):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    adapter = make_adapter(tmp_path, "run_a")

    with pytest.raises(SystemExit) as exit_info:
        quality.main([
            "--prompts", str(prompt_file),
            "--from-report", str(report),
            "--adapter", str(adapter),
        ])

    assert exit_info.value.code == 2
    assert "--adapter cannot be combined with --from-report" in capsys.readouterr().err


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        quality.main(["--help"])

    assert exit_info.value.code == 0
    assert "--from-report" in capsys.readouterr().out
