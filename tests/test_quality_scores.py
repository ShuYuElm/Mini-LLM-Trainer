"""Review workbook import: validation, blinding, and score-sheet writing."""

import json
from pathlib import Path

import pytest

import generate_quality_eval as quality
import import_quality_scores as scores
from audit_quality_eval import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = ["base", "lora_r8_seed42_baseline_v1", "lora_r32_seed42_baseline_v1"]


@pytest.fixture
def payload():
    return json.loads((ROOT / "configs/quality_eval_v1.json").read_text(encoding="utf-8"))


def sample_report(payload, run_ids=RUN_IDS, count=3):
    """Responses carry no run identity, so exported text can be checked for leaks."""
    prompts = payload["prompts"][:count]
    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_count": count,
        "complete_prompt_set": count == len(payload["prompts"]),
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "runs": [
            {
                "run_id": run_id,
                "samples": [
                    {"id": prompt["id"], "response": f"reply {position} to {prompt['id']}"}
                    for prompt in prompts
                ],
            }
            for position, run_id in enumerate(run_ids, start=1)
        ],
    }


def write_inputs(tmp_path, payload, report, seed=11):
    workbook = tmp_path / "workbook.md"
    workbook.write_text(quality.build_review_workbook(payload, report, seed), encoding="utf-8")

    sheet = tmp_path / "scores.json"
    sheet.write_text(scores.dump_json(quality.build_score_sheet(payload, report)), encoding="utf-8")

    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    return workbook, sheet, prompts


def fill(text, score="2 1 2 1", rationale="checked"):
    return scores.SCORE_LINE.sub(lambda match: f"{match.group('label')}: {score} | {rationale}", text)


def run_import(workbook, sheet, prompts):
    return scores.main([
        "--workbook", str(workbook),
        "--scores", str(sheet),
        "--prompts", str(prompts),
    ])


def test_import_records_every_score(tmp_path, payload):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(fill(workbook.read_text(encoding="utf-8")), encoding="utf-8")

    run_import(workbook, sheet, prompts)

    result = json.loads(sheet.read_text(encoding="utf-8"))

    assert len(result["entries"]) == 9
    assert result["prompt_set_sha256"] == canonical_hash(payload)
    assert result["entries"][0]["prompt_id"] == report["prompt_ids"][0]
    assert [entry["label"] for entry in result["entries"][:3]] == ["A", "B", "C"]

    for entry in result["entries"]:
        assert entry["scores"] == {
            "correctness": 2, "completeness": 1, "format_compliance": 2, "no_unsupported_additions": 1,
        }
        assert entry["rationale"] == "checked"


def test_import_summary_counts_scores(tmp_path, payload, capsys):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(fill(workbook.read_text(encoding="utf-8"), score="0 0 0 0"), encoding="utf-8")

    run_import(workbook, sheet, prompts)

    printed = capsys.readouterr().out

    assert "recorded 9 scores for 3 prompts" in printed
    assert "correctness: 0=9 1=0 2=0" in printed


def test_unfilled_workbook_writes_nothing(tmp_path, payload, capsys):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    before = sheet.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run_import(workbook, sheet, prompts)

    printed = capsys.readouterr().out

    assert exit_info.value.code == 1
    assert "not scored yet" in printed
    assert "nothing written" in printed
    assert sheet.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("score, rationale, message", [
    ("3 1 2 1", "checked", "'3' is not 0, 1, or 2"),
    ("2 1 2", "checked", "expected 4 scores, found 3"),
    ("2 1 2 1", "", "empty rationale"),
])
def test_invalid_score_lines_are_reported(tmp_path, payload, capsys, score, rationale, message):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(
        fill(workbook.read_text(encoding="utf-8"), score=score, rationale=rationale),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        run_import(workbook, sheet, prompts)

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().out


def test_duplicate_score_line_is_rejected():
    text = "# 手册\n<!-- prompt_set_sha256: " + "a" * 64 + " -->\n\n## 1. `p`\n\nA: 1 1 1 1 | x\nA: 2 2 2 2 | y\n"

    with pytest.raises(ValueError, match="duplicate score line"):
        scores.parse_workbook(text)


def test_missing_hash_marker_is_rejected():
    with pytest.raises(ValueError, match="hash marker"):
        scores.parse_workbook("## 1. `p`\n\nA: 1 1 1 1 | x\n")


def test_workbook_from_another_prompt_set_is_rejected(tmp_path, payload, capsys):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    text = fill(workbook.read_text(encoding="utf-8"))
    workbook.write_text(text.replace(canonical_hash(payload), "0" * 64), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run_import(workbook, sheet, prompts)

    assert exit_info.value.code == 1
    assert "does not match the prompt set" in capsys.readouterr().out


def test_unknown_prompt_in_workbook_is_rejected(tmp_path, payload, capsys):
    report = sample_report(payload, count=3)
    workbook, _, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(fill(workbook.read_text(encoding="utf-8")), encoding="utf-8")

    smaller = sample_report(payload, count=2)
    sheet = tmp_path / "smaller_scores.json"
    sheet.write_text(scores.dump_json(quality.build_score_sheet(payload, smaller)), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run_import(workbook, sheet, prompts)

    assert exit_info.value.code == 1
    assert "not part of this score sheet" in capsys.readouterr().out


def test_repeated_import_replaces_previous_scores(tmp_path, payload, capsys):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(fill(workbook.read_text(encoding="utf-8")), encoding="utf-8")

    run_import(workbook, sheet, prompts)
    capsys.readouterr()

    workbook.write_text(fill(workbook.read_text(encoding="utf-8"), score="1 1 1 0"), encoding="utf-8")
    run_import(workbook, sheet, prompts)

    assert "replaced 9 previously recorded scores" in capsys.readouterr().out
    assert json.loads(sheet.read_text(encoding="utf-8"))["entries"][0]["scores"]["correctness"] == 1


def test_workbook_never_contains_run_identities(payload):
    text = quality.build_review_workbook(payload, sample_report(payload), 11)

    for run_id in RUN_IDS:
        assert run_id not in text

    assert '"base"' not in text


def test_score_sheet_dimension_order_is_validated(payload):
    report = sample_report(payload)
    sheet = quality.build_score_sheet(payload, report)
    sheet["entries"][0]["scores"] = {
        "completeness": None, "correctness": None, "format_compliance": None, "no_unsupported_additions": None,
    }

    with pytest.raises(ValueError, match="dimensions differ"):
        scores.fill_sheet(sheet, {})


def test_import_does_not_read_the_mapping_file(tmp_path, payload, monkeypatch):
    report = sample_report(payload)
    workbook, sheet, prompts = write_inputs(tmp_path, payload, report)
    workbook.write_text(fill(workbook.read_text(encoding="utf-8")), encoding="utf-8")

    mapped = tmp_path / quality.REVIEW_MAPPING
    mapped.write_text('{"labels": {}}', encoding="utf-8")

    original = Path.read_text

    def guarded(self, *args, **kwargs):
        if self.name == quality.REVIEW_MAPPING:
            raise AssertionError("the mapping must not be read while importing scores")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)

    run_import(workbook, sheet, prompts)

    result = json.loads(original(sheet, encoding="utf-8"))

    assert all(entry["rationale"] == "checked" for entry in result["entries"])
