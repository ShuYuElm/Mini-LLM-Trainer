"""Prompt integrity and overlap heuristics; no model or local data required."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from audit_quality_eval import (
    canonical_hash, check_prompt_lengths, find_overlaps, main, normalize,
    validate_prompt_set,
)


@pytest.fixture
def payload():
    path = Path(__file__).resolve().parents[1] / "configs/quality_eval_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_repository_prompt_set_has_40_balanced_tasks_and_16_checkable_references(payload):
    before = deepcopy(payload)
    validate_prompt_set(payload)
    assert payload == before
    assert sum(p["automated_check"]["type"] != "human_only" for p in payload["prompts"]) == 16


@pytest.mark.parametrize("case, message", [
    ("count", "40 prompts"), ("id", "Duplicate ID"),
    ("duplicate_text", "duplicate normalized"), ("category", "eight prompts"),
    ("facts", "required_facts"), ("reference", "reference_answer"),
    ("rubric", "scores must"), ("format", "format_requirement"),
    ("generation", "generation plan"), ("json_type", "values/types"),
    ("lines", "line reference mismatch"),
])
def test_invalid_prompt_set_is_rejected(payload, case, message):
    first = payload["prompts"][0]
    if case == "count":
        payload["prompts"].pop()
    elif case == "id":
        payload["prompts"][1]["id"] = first["id"]
    elif case == "duplicate_text":
        payload["prompts"][1].update(instruction=first["instruction"].upper(), input=first["input"])
    elif case == "category":
        first["category"] = "reasoning"
    elif case == "facts":
        first["required_facts"] = []
    elif case == "reference":
        first["reference_answer"] = " "
    elif case == "rubric":
        del payload["rubric"]["correctness"]["2"]
    elif case == "format":
        first["format_requirement"] = ""
    elif case == "generation":
        payload["generation_plan"]["do_sample"] = True
    elif case == "json_type":
        target = next(p for p in payload["prompts"] if p["id"] == "extract_boolean")
        target["automated_check"]["expected"]["inspection_passed"] = 1
    elif case == "lines":
        target = next(p for p in payload["prompts"] if p["id"] == "format_csv")
        target["reference_answer"] = "name,score\nTalia,85"
    with pytest.raises(ValueError, match=message):
        validate_prompt_set(payload)


def test_hash_is_stable_under_json_whitespace_and_key_order(payload):
    restored = json.loads(json.dumps(payload, ensure_ascii=True, indent=4))
    assert canonical_hash(payload) == canonical_hash(restored)
    changed = deepcopy(payload)
    changed["prompts"][0]["required_facts"].append("A changed grading criterion")
    assert canonical_hash(changed) != canonical_hash(payload)


def test_normalization_handles_unicode_case_and_punctuation():
    assert normalize("ＡＢＣ  Café,  HELLO!\nworld") == "abc café hello world"


def corpus_row(instruction, text):
    return {"source": "train", "source_id": 7, "instruction": instruction, "input": text}


def test_exact_prompt_overlap_survives_case_and_punctuation_changes():
    prompts = [{"id": "one", "instruction": "Return a label.", "input": "alpha"}]
    matches = find_overlaps(prompts, [corpus_row("RETURN A LABEL!", "Alpha")])
    assert len(matches) == 1
    assert matches[0]["kind"] == "normalized_exact"
    assert matches[0]["source_id"] == 7


def test_identical_long_input_detected_with_different_instruction():
    text = "Twelve parcels arrived at the west depot before the morning inspection began."
    matches = find_overlaps(
        [{"id": "one", "instruction": "Summarize the text.", "input": text}],
        [corpus_row("Extract the location.", text)],
    )
    assert any(m["field"] == "input" and m["kind"] == "normalized_exact" for m in matches)


def test_near_match_detects_small_edit_without_marking_it_exact():
    original = "Twelve parcels arrived at the west depot before the morning inspection began and were stored beside the loading door."
    edited = original.replace("Twelve", "Thirteen")
    matches = find_overlaps(
        [{"id": "one", "instruction": "Summarize.", "input": edited}],
        [corpus_row("Summarize.", original)],
    )
    assert matches and all(m["kind"] == "near_lexical" for m in matches)
    assert all(m["shared_trigrams"] >= 8 for m in matches)


def test_shared_short_instruction_does_not_alone_trigger_overlap():
    assert find_overlaps(
        [{"id": "one", "instruction": "Answer briefly.", "input": "orchid"}],
        [corpus_row("Answer briefly.", "granite")],
    ) == []


def test_token_length_check_uses_only_inference_input(payload):
    tokenizer = Mock(return_value={"input_ids": [1] * 10})
    prompt = deepcopy(payload["prompts"][0])
    prompt["reference_answer"] = "SECRET_REFERENCE"
    prompt["required_facts"] = ["SECRET_CRITERIA"]
    assert check_prompt_lengths([prompt], tokenizer, 10) == {prompt["id"]: 10}
    text = tokenizer.call_args.args[0]
    assert text.endswith("### Response:\n")
    assert prompt["instruction"] in text and prompt["input"] in text
    assert "SECRET" not in text
    with pytest.raises(ValueError, match="exceeds input limit"):
        check_prompt_lengths([prompt], tokenizer, 9)


@pytest.mark.parametrize("overlap", [False, True])
def test_audit_cli_writes_counts_fingerprints_and_match_status(payload, tmp_path, monkeypatch, overlap):
    import datasets
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(payload), encoding="utf-8")
    development = tmp_path / "dev.json"
    development.write_text("[]", encoding="utf-8")
    splits = {name: [{"sample_id": i, "instruction": "Unrelated", "input": "", "output": "unrelated"} for i in range(size)]
              for name, size in (("train", 2000), ("validation", 200))}
    if overlap:
        splits["train"][0].update(instruction=payload["prompts"][0]["instruction"], input=payload["prompts"][0]["input"])
    loader = Mock(return_value=splits)
    monkeypatch.setattr(datasets, "load_from_disk", loader)
    output = tmp_path / "audit.json"
    main(["--prompts", str(prompt_path), "--data-dir", str(tmp_path / "data"),
          "--development-prompts", str(development), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["prompt_set_sha256"] == canonical_hash(payload)
    assert result["sources"]["counts"] == {"train": 2000, "validation": 200}
    assert result["sources"]["split_sha256"]["train"] == canonical_hash(splits["train"])
    assert result["status"] == ("needs_review" if overlap else "no_lexical_matches")
    assert bool(result["matches"]) == overlap
    assert result["input_token_lengths"] is None
    loader.assert_called_once_with(str(tmp_path / "data"))
