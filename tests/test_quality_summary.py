"""Post-review joins and descriptive statistics use synthetic ratings only."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from audit_quality_eval import canonical_hash
from import_quality_scores import DIMENSIONS
from summarize_quality_scores import aggregate, main


@pytest.fixture
def documents():
    payload = json.loads((Path(__file__).resolve().parents[1] / "configs/quality_eval_v1.json").read_text(encoding="utf-8"))
    common = {"evaluation_id": payload["evaluation_id"], "prompt_set_sha256": canonical_hash(payload), "complete_prompt_set": True}
    sheet = {**common, "rubric": deepcopy(payload["rubric"]), "entries": []}
    mapping = {**common, "review_seed": 7, "labels": {}}
    generation = {**common, "runs": [{"run_id": run, "samples": []} for run in ("base", "r8", "r32")]}
    blind = {**common, "review_seed": 7, "answers": []}
    runs = [run["run_id"] for run in generation["runs"]]
    for i, prompt in enumerate(payload["prompts"]):
        pid = prompt["id"]
        order = runs[i % 3:] + runs[:i % 3]
        mapping["labels"][pid] = dict(zip("ABC", order))
        candidates = []
        for label, run in mapping["labels"][pid].items():
            value = runs.index(run)
            sheet["entries"].append({"prompt_id": pid, "category": prompt["category"], "label": label,
                                     "scores": dict.fromkeys(DIMENSIONS, value), "rationale": "Synthetic rating"})
            candidates.append({"label": label, "response": f"{pid}: {run} response"})
        blind["answers"].append({"prompt_id": pid, "candidates": candidates})
        for run in generation["runs"]:
            run["samples"].append({"id": pid, "response": f"{pid}: {run['run_id']} response"})
    return payload, sheet, mapping, generation, blind


def test_per_prompt_mapping_drives_means_categories_and_paired_counts(documents):
    before = deepcopy(documents)
    result = aggregate(*documents)
    assert documents == before
    assert len(result["entries"]) == 120 and len(result["groups"]) == 18
    for row in result["groups"]:
        value = {"base": 0, "r8": 1, "r32": 2}[row["run_id"]]
        assert row["n"] == (40 if row["category"] == "all" else 8)
        assert all(row[d] == value for d in DIMENSIONS)
        assert row["mean_total_out_of_8"] == value * 4
    for pair in result["paired_total_comparisons"]:
        assert pair["n"] == pair["left_lower"] == 40
        assert pair["left_higher"] == pair["tied"] == 0
    assert result["identical_response_score_inconsistencies"] == []


@pytest.mark.parametrize("case", [
    "missing_score", "boolean_score", "missing_entry", "duplicate_entry", "wrong_category",
    "wrong_hash", "partial", "duplicate_mapping", "wrong_blind_text", "missing_blind",
    "wrong_rubric", "empty_rationale",
])
def test_invalid_or_unfinished_inputs_cannot_be_aggregated(documents, case):
    payload, sheet, mapping, generation, blind = documents
    if case == "missing_score":
        sheet["entries"][0]["scores"]["correctness"] = None
    elif case == "boolean_score":
        sheet["entries"][0]["scores"]["correctness"] = True
    elif case == "missing_entry":
        sheet["entries"].pop()
    elif case == "duplicate_entry":
        sheet["entries"].append(deepcopy(sheet["entries"][0]))
    elif case == "wrong_category":
        sheet["entries"][0]["category"] = "wrong"
    elif case == "wrong_hash":
        mapping["prompt_set_sha256"] = "0" * 64
    elif case == "partial":
        generation["complete_prompt_set"] = False
    elif case == "duplicate_mapping":
        mapping["labels"][payload["prompts"][0]["id"]]["C"] = "base"
    elif case == "wrong_blind_text":
        blind["answers"][0]["candidates"][0]["response"] = "unrelated"
    elif case == "missing_blind":
        blind["answers"][0]["candidates"].pop()
    elif case == "wrong_rubric":
        sheet["rubric"]["correctness"]["2"] = "Changed anchor"
    elif case == "empty_rationale":
        sheet["entries"][0]["rationale"] = " "
    with pytest.raises(ValueError):
        aggregate(*documents)


def test_identical_response_rating_disagreement_is_flagged_not_corrected(documents):
    _, sheet, _, generation, blind = documents
    response = generation["runs"][0]["samples"][0]["response"]
    generation["runs"][1]["samples"][0]["response"] = response
    blind["answers"][0]["candidates"][1]["response"] = response
    before = deepcopy(sheet)
    result = aggregate(*documents)
    assert len(result["identical_response_score_inconsistencies"]) == 1
    assert sheet == before


@pytest.mark.parametrize("valid", [False, True])
def test_cli_preserves_source_files_and_only_writes_after_validation(documents, tmp_path, valid):
    payload, *inputs = documents
    if not valid:
        inputs[0]["entries"][0]["scores"]["correctness"] = None
    names = ("quality_review_scores.json", "quality_review_mapping.json", "quality_generation.json", "quality_review_blind.json")
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(payload), encoding="utf-8")
    for name, data in zip(names, inputs):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    before = {name: (tmp_path / name).read_bytes() for name in names}
    args = ["--review-dir", str(tmp_path), "--prompts", str(prompt_path)]
    if valid:
        main(args)
        with (tmp_path / "quality_score_details.csv").open(encoding="utf-8-sig", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 120
        assert (tmp_path / "quality_score_summary.md").is_file()
    else:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
        assert not (tmp_path / "quality_score_summary.json").exists()
        assert list(tmp_path.glob("*.csv")) == []
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
