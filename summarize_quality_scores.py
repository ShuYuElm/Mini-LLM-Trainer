"""Reveal completed blind scores and export descriptive model comparisons.

Reads saved JSON only. Never changes ratings or generates model responses.
"""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

from audit_quality_eval import canonical_hash, validate_prompt_set
from import_quality_scores import DIMENSIONS


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "experiments/quality_eval_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(payload, sheet, mapping, generation, blind):
    validate_prompt_set(payload)
    expected_hash = canonical_hash(payload)
    for name, doc in (("scores", sheet), ("mapping", mapping), ("generation", generation), ("blind", blind)):
        require(doc["evaluation_id"] == payload["evaluation_id"] and doc["prompt_set_sha256"] == expected_hash, f"{name}: prompt-set mismatch")
    for name, doc in (("scores", sheet), ("generation", generation), ("blind", blind)):
        require(doc["complete_prompt_set"] is True, f"{name}: incomplete prompt set")
    require(sheet["rubric"] == payload["rubric"], "score rubric mismatch")
    require(blind["review_seed"] == mapping["review_seed"], "review seed mismatch")
    prompts = {p["id"]: p for p in payload["prompts"]}
    require(set(mapping["labels"]) == set(prompts), "mapping prompt coverage mismatch")
    runs = [run["run_id"] for run in generation["runs"]]
    require(len(runs) == len(set(runs)) == 3 and "base" in runs, "expected three unique runs including base")
    responses = {}
    for run in generation["runs"]:
        require(len(run["samples"]) == len(prompts), "response count mismatch")
        indexed = {s["id"]: s["response"] for s in run["samples"]}
        require(set(indexed) == set(prompts), "response prompt coverage mismatch")
        responses[run["run_id"]] = indexed
    assignments = {}
    for pid, labels in mapping["labels"].items():
        require(set(labels) == {"A", "B", "C"} and set(labels.values()) == set(runs), f"{pid}: mapping must assign each run once")
        assignments.update({(pid, label): run for label, run in labels.items()})
    require(len(blind["answers"]) == len(prompts), "blind prompt count mismatch")
    seen_blind = set()
    for answer in blind["answers"]:
        pid = answer["prompt_id"]
        require(pid in prompts, "unknown blind prompt")
        for candidate in answer["candidates"]:
            key = (pid, candidate["label"])
            require(key in assignments and key not in seen_blind, "blind candidate mismatch")
            require(candidate["response"] == responses[assignments[key]][pid], f"{pid}: blind response does not match mapping")
            seen_blind.add(key)
    require(seen_blind == set(assignments), "blind candidate coverage mismatch")

    entries, seen = [], set()
    for entry in sheet["entries"]:
        key = (entry["prompt_id"], entry["label"])
        require(key in assignments and key not in seen, "duplicate or unknown score entry")
        seen.add(key)
        require(entry["category"] == prompts[key[0]]["category"], "score category mismatch")
        scores = entry["scores"]
        require(set(scores) == set(DIMENSIONS), "score dimensions mismatch")
        require(all(type(v) is int and v in (0, 1, 2) for v in scores.values()), f"{key}: missing or invalid score")
        require(isinstance(entry["rationale"], str) and entry["rationale"].strip(), f"{key}: empty rationale")
        entries.append({"prompt_id": key[0], "category": entry["category"], "label": key[1],
                        "run_id": assignments[key], **scores, "total": sum(scores.values()), "rationale": entry["rationale"]})
    require(seen == set(assignments), "incomplete score coverage")

    categories = list(dict.fromkeys(p["category"] for p in payload["prompts"]))
    groups = []
    for run in runs:
        for category in ["all", *categories]:
            selected = [e for e in entries if e["run_id"] == run and (category == "all" or e["category"] == category)]
            groups.append({"run_id": run, "category": category, "n": len(selected),
                           **{d: sum(e[d] for e in selected) / len(selected) for d in DIMENSIONS},
                           "mean_total_out_of_8": sum(e["total"] for e in selected) / len(selected)})
    totals = {(e["run_id"], e["prompt_id"]): e["total"] for e in entries}
    pairs = []
    for left, right in combinations(runs, 2):
        differences = [totals[(left, pid)] - totals[(right, pid)] for pid in prompts]
        pairs.append({"left": left, "right": right, "n": len(differences),
                      "left_higher": sum(d > 0 for d in differences), "tied": differences.count(0),
                      "left_lower": sum(d < 0 for d in differences)})
    identical = defaultdict(list)
    for entry in entries:
        response = responses[entry["run_id"]][entry["prompt_id"]].replace("\r\n", "\n").strip()
        identical[(entry["prompt_id"], response)].append(entry)
    inconsistencies = []
    for (pid, _), selected in identical.items():
        ratings = {tuple(e[d] for d in DIMENSIONS) for e in selected}
        if len(selected) > 1 and len(ratings) > 1:
            inconsistencies.append({"prompt_id": pid, "labels": [e["label"] for e in selected]})
    return {
        "evaluation_id": payload["evaluation_id"], "prompt_set_sha256": expected_hash,
        "source_canonical_sha256": {name: canonical_hash(doc) for name, doc in
                                    (("scores", sheet), ("mapping", mapping), ("generation", generation), ("blind", blind))},
        "rating_source": "User-completed anonymous workbook; original ratings retained without adjudication.",
        "groups": groups, "paired_total_comparisons": pairs, "entries": entries,
        "identical_response_score_inconsistencies": inconsistencies,
        "limitations": [
            "One reviewer, 40 synthetic prompts (8 per category), one generation per model/prompt.",
            "Ordinal 0/1/2 dimension means are descriptive; total out of 8 is a secondary equal-weight summary.",
            "Dimensions overlap and are not independent. Paired totals are not direct preference judgments.",
            "No inter-rater reliability or statistical significance is established.",
            "These scores do not establish general model superiority or a percentage quality improvement.",
        ],
    }


def markdown_report(summary):
    lines = ["# 回答质量评分汇总", "", "评分来源：用户完成的匿名评分手册；原始分数和理由保持不变。", "",
             "每个维度为 0–2 分；总分为四维等权相加，满分 8 分，仅作辅助描述。", "",
             "| 模型 | 类别 | 回答数 | 正确性 | 完整性 | 格式 | 无无据添加 | 总分均值 /8 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in summary["groups"]:
        means = " | ".join(f"{group[d]:.3f}" for d in DIMENSIONS)
        lines.append(f"| {group['run_id']} | {group['category']} | {group['n']} | {means} | {group['mean_total_out_of_8']:.3f} |")
    lines += ["", "同一题上的总分比较：数字按左侧模型总分较高 / 相同 / 较低列出，不代表人工两两偏好投票。", ""]
    for pair in summary["paired_total_comparisons"]:
        lines.append(f"- {pair['left']} vs {pair['right']}: {pair['left_higher']} / {pair['tied']} / {pair['left_lower']}（共 {pair['n']} 题）")
    lines += ["", f"相同回答但不同评分的组数：{len(summary['identical_response_score_inconsistencies'])}。", "",
              "本结果仅覆盖一名评分者、40 道合成题和每模型每题一次生成。没有多评分者一致性或统计显著性证据，不能推断一般能力优劣。", "",
              "- [分模型与类别 CSV](quality_score_summary.csv)",
              "- [逐题分数与理由 CSV](quality_score_details.csv)",
              "- [汇总 JSON 与来源哈希](quality_score_summary.json)", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/quality_eval_v1.json")
    args = parser.parse_args(argv)
    names = ("quality_review_scores.json", "quality_review_mapping.json", "quality_generation.json", "quality_review_blind.json")
    try:
        payload = json.loads(args.prompts.read_text(encoding="utf-8"))
        documents = [json.loads((args.review_dir / name).read_text(encoding="utf-8")) for name in names]
        summary = aggregate(payload, *documents)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for name, rows in (("quality_score_summary.csv", summary["groups"]), ("quality_score_details.csv", summary["entries"])):
        with (args.review_dir / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.review_dir / "quality_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.review_dir / "quality_score_summary.md").write_text(markdown_report(summary), encoding="utf-8")
    for group in summary["groups"]:
        if group["category"] == "all":
            print(f"{group['run_id']}: {group['mean_total_out_of_8']:.3f}/8 (n={group['n']})")
    print(f"Saved summary and detailed scores in {args.review_dir}")


if __name__ == "__main__":
    main()
