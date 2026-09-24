"""Generate held-out quality-eval responses and export anonymous review files."""

import argparse
import json
import random
import re
import time
from pathlib import Path

from audit_quality_eval import canonical_hash, validate_prompt_set
from src.evaluation import generate_response
from src.utils import save_json


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = ROOT / "model"
DEFAULT_PROMPTS = ROOT / "configs" / "quality_eval_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "quality_eval_v1"

# The adapter run identifier comes from its run directory; base has no adapter.
BASE_RUN_ID = "base"
CANDIDATE_LABELS = "ABCDEFGH"

GENERATION_REPORT = "quality_generation.json"
BLIND_REVIEW = "quality_review_blind.json"
REVIEW_MAPPING = "quality_review_mapping.json"
SCORE_SHEET = "quality_review_scores.json"
WORKBOOK = "quality_review_workbook.md"
REVIEW_FILES = (BLIND_REVIEW, REVIEW_MAPPING, SCORE_SHEET, WORKBOOK)

CATEGORY_LABELS = {
    "summarization": "摘要",
    "rewriting": "改写",
    "extraction": "信息提取",
    "format_following": "格式遵循",
    "reasoning": "简单推理",
}

DIMENSION_LABELS = {
    "correctness": "正确性",
    "completeness": "完整性",
    "format_compliance": "格式遵循",
    "no_unsupported_additions": "无无据添加",
}

LIMITATIONS = (
    "Forty project-authored synthetic tasks are not an external benchmark.",
    "Deterministic checks cover only the stated format property on 16 of 40 tasks; the other 24 tasks need human review.",
    "References and criteria were authored together with the prompts and are not authoritative labels.",
    "These responses do not establish general model quality; human scores are recorded separately.",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Quality prompt set with references and scoring criteria.",
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        action="append",
        default=[],
        help="Saved LoRA adapter to evaluate; repeat for several runs. The base model is always included.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generation report and the review files.",
    )

    parser.add_argument(
        "--from-report",
        type=Path,
        default=None,
        help="Re-export review files from an existing report without loading a model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260921,
        help="Seed for the per-prompt anonymous answer order.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N prompts; used for smoke checks.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of refusing to run.",
    )

    args = parser.parse_args(argv)

    if not args.prompts.is_file():
        parser.error(f"Prompt set does not exist: {args.prompts}")

    if args.from_report is not None:
        if not args.from_report.is_file():
            parser.error(f"Report does not exist: {args.from_report}")
        if args.adapter:
            parser.error("--adapter cannot be combined with --from-report")

    for adapter in args.adapter:
        if not adapter.is_file():
            parser.error(f"Adapter file does not exist: {adapter}")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32)")

    return args


def run_specs(adapters):
    """Base first, then one run per adapter, named after its run directory."""
    specs = [{"run_id": BASE_RUN_ID, "adapter": None}]

    for adapter in adapters:
        specs.append({"run_id": adapter.parent.name, "adapter": adapter})

    run_ids = [spec["run_id"] for spec in specs]
    duplicates = sorted({name for name in run_ids if run_ids.count(name) > 1})

    if duplicates:
        raise ValueError(f"duplicate run ids from adapter directories: {duplicates}")

    if len(run_ids) > len(CANDIDATE_LABELS):
        raise ValueError("more runs than available candidate labels")

    return specs


def protect_outputs(output_dir, names, overwrite):
    existing = [output_dir / name for name in names if (output_dir / name).exists()]

    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"existing output files: {listed} (pass --overwrite to replace them)")


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )


def load_run_model(adapter_path, device):
    """Load the base model and, when given, inject and load one adapter into it."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        dtype=torch.bfloat16,
    )

    model.to(device)

    metadata = None

    if adapter_path is not None:
        # Reuse the evaluation loader so both entry points check adapters the same way.
        from evaluate import load_adapter_for_evaluation

        metadata = load_adapter_for_evaluation(
            model=model,
            adapter_path=adapter_path,
            base_model_name=MODEL_NAME,
        )

    return model, metadata


def generate_run_responses(model, tokenizer, prompts, plan, device):
    """One greedy response per prompt; only instruction and input reach the model."""
    inputs = [
        {
            "id": prompt["id"],
            "instruction": prompt["instruction"],
            "input": prompt["input"],
        }
        for prompt in prompts
    ]

    samples = []

    for item in inputs:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            instruction=item["instruction"],
            input_text=item["input"],
            device=device,
            max_input_length=plan["max_input_length"],
            max_new_tokens=plan["max_new_tokens"],
            use_amp=True,
        )

        samples.append({"id": item["id"], "response": response})
        print("generated:", item["id"])

    return samples


def check_json_object(response, expected):
    """One bare JSON object with exactly the expected keys, values, and value types."""
    text = response.strip()

    if not text:
        return False, "empty response"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False, "response is not a single JSON value"

    if not isinstance(parsed, dict):
        return False, "response is not a JSON object"

    if canonical_hash(parsed) != canonical_hash(expected):
        return False, "keys, values, or value types differ from the expected object"

    return True, ""


def check_exact_lines(response, expected):
    """Outer whitespace and newline style may differ; line text and order may not."""
    text = response.strip()

    if not text:
        return False, "empty response"

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    if len(lines) != len(expected):
        return False, f"expected {len(expected)} lines, found {len(lines)}"

    for index, (line, wanted) in enumerate(zip(lines, expected), start=1):
        if line != wanted:
            return False, f"line {index} differs"

    return True, ""


CHECKS = {"json_object": check_json_object, "exact_lines": check_exact_lines}


def run_automated_checks(prompts, records):
    """Deterministic format checks; they never replace the four human ratings."""
    results = {
        "method": "Deterministic checks measure only their stated format property and do not replace the four human ratings.",
        "checks": {},
    }

    for kind, check in CHECKS.items():
        tasks = [prompt for prompt in prompts if prompt["automated_check"]["type"] == kind]

        if not tasks:
            continue

        responses = {
            record["run_id"]: {sample["id"]: sample["response"] for sample in record["samples"]}
            for record in records
        }

        passed = 0
        failures = []

        for run_id, samples in responses.items():
            for task in tasks:
                ok, reason = check(samples[task["id"]], task["automated_check"]["expected"])

                if ok:
                    passed += 1
                else:
                    failures.append({"run_id": run_id, "prompt_id": task["id"], "reason": reason})

        results["checks"][kind] = {
            "tasks": len(tasks),
            "answers": len(tasks) * len(records),
            "passed": passed,
            "failures": failures,
        }

    return results


def build_report(payload, prompts, records, device):
    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_set_sha256": canonical_hash(payload),
        "prompt_count": len(prompts),
        "prompt_set_prompt_count": len(payload["prompts"]),
        "complete_prompt_set": len(prompts) == len(payload["prompts"]),
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "model_name": MODEL_NAME,
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "generation": {
            "max_input_length": payload["generation_plan"]["max_input_length"],
            "max_new_tokens": payload["generation_plan"]["max_new_tokens"],
            "do_sample": payload["generation_plan"]["do_sample"],
            "device": str(device),
            "dtype": "bfloat16",
            "use_amp": True,
        },
        "runs": records,
        "automated_checks": run_automated_checks(prompts, records),
        "limitations": list(LIMITATIONS),
    }


def generate_report(payload, specs, limit):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_prompts = payload["prompts"]
    prompts = list(all_prompts[:limit]) if limit is not None else list(all_prompts)
    plan = payload["generation_plan"]

    tokenizer = load_tokenizer()
    records = []

    for spec in specs:
        print("loading model for run:", spec["run_id"])

        model, metadata = load_run_model(spec["adapter"], device)

        start = time.perf_counter()
        samples = generate_run_responses(model, tokenizer, prompts, plan, device)
        seconds = time.perf_counter() - start

        records.append({
            "run_id": spec["run_id"],
            "adapter_path": str(spec["adapter"]) if spec["adapter"] is not None else None,
            "adapter_metadata": metadata,
            "responses": len(samples),
            "generation_seconds": seconds,
            "samples": samples,
        })

        print(f"run {spec['run_id']}: {len(samples)} responses in {seconds:.1f} seconds")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return build_report(payload, prompts, records, device)


def report_run_ids(report):
    return [record["run_id"] for record in report["runs"]]


def report_prompts(payload, report):
    by_id = {prompt["id"]: prompt for prompt in payload["prompts"]}
    missing = [prompt_id for prompt_id in report["prompt_ids"] if prompt_id not in by_id]

    if missing:
        raise ValueError(f"report references unknown prompts: {missing[:3]}")

    return [by_id[prompt_id] for prompt_id in report["prompt_ids"]]


def indexed_responses(report, prompt_ids):
    indexed = {}

    for record in report["runs"]:
        samples = {sample["id"]: sample["response"] for sample in record["samples"]}
        missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in samples]

        if missing:
            raise ValueError(f"run {record['run_id']} has no response for {len(missing)} prompts, first: {missing[0]}")

        indexed[record["run_id"]] = samples

    return indexed


def shuffled_run_ids(prompt_id, run_ids, seed):
    order = list(run_ids)
    random.Random(f"{seed}:{prompt_id}").shuffle(order)
    return order


def blind_assignments(prompt_ids, run_ids, seed):
    return {
        prompt_id: dict(zip(CANDIDATE_LABELS, shuffled_run_ids(prompt_id, run_ids, seed)))
        for prompt_id in prompt_ids
    }


def build_blind_review(payload, report, seed):
    """Answers only, in a shuffled per-prompt order and without model identities."""
    prompt_ids = report["prompt_ids"]
    run_ids = report_run_ids(report)
    assignments = blind_assignments(prompt_ids, run_ids, seed)
    responses = indexed_responses(report, prompt_ids)

    answers = []

    for prompt in report_prompts(payload, report):
        prompt_id = prompt["id"]

        answers.append({
            "prompt_id": prompt_id,
            "category": prompt["category"],
            "instruction": prompt["instruction"],
            "input": prompt["input"],
            "reference_answer": prompt["reference_answer"],
            "required_facts": list(prompt["required_facts"]),
            "format_requirement": prompt["format_requirement"],
            "prohibited_additions": list(prompt["prohibited_additions"]),
            "candidates": [
                {"label": label, "response": responses[run_id][prompt_id]}
                for label, run_id in assignments[prompt_id].items()
            ],
        })

    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_set_sha256": canonical_hash(payload),
        "review_seed": seed,
        "complete_prompt_set": report["complete_prompt_set"],
        "instructions": "Score every candidate without seeing model identities. The label-to-run mapping is kept in a separate file and revealed after scoring.",
        "review_protocol": list(payload["review_protocol"]),
        "rubric": payload["rubric"],
        "answers": answers,
    }


def build_review_mapping(payload, report, seed):
    prompt_ids = report["prompt_ids"]
    run_ids = report_run_ids(report)

    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_set_sha256": canonical_hash(payload),
        "review_seed": seed,
        "instructions": "Keep this mapping away from the scoring file until every score is recorded.",
        "labels": blind_assignments(prompt_ids, run_ids, seed),
    }


def build_score_sheet(payload, report):
    dimensions = list(payload["rubric"])
    labels = CANDIDATE_LABELS[: len(report_run_ids(report))]
    entries = []

    for prompt in report_prompts(payload, report):
        for label in labels:
            entries.append({
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "label": label,
                "scores": {dimension: None for dimension in dimensions},
                "rationale": "",
            })

    return {
        "evaluation_id": payload["evaluation_id"],
        "prompt_set_sha256": canonical_hash(payload),
        "complete_prompt_set": report["complete_prompt_set"],
        "instructions": "Record an integer 0, 1, or 2 and a short rationale for each dimension. Score from the blind file only, then reveal the mapping.",
        "rubric": payload["rubric"],
        "entries": entries,
    }


def fenced(text):
    """Wrap text in a code fence long enough to survive its own backticks."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def table_cell(text):
    return text.replace("|", "\\|")


def build_review_workbook(payload, report, seed):
    """Human-readable scoring workbook with the same anonymity as the blind file."""
    blind = build_blind_review(payload, report, seed)
    dimensions = list(payload["rubric"])
    answers = blind["answers"]
    candidates_per_prompt = len(answers[0]["candidates"]) if answers else 0
    order = "、".join(DIMENSION_LABELS.get(name, name) for name in dimensions)

    lines = [
        "# 独立回答质量评估 · 评分手册",
        "",
        f"<!-- prompt_set_sha256: {blind['prompt_set_sha256']} -->",
        "",
        f"- 题集哈希：`{blind['prompt_set_sha256']}`",
        f"- 题目数：{len(answers)}；每题候选数：{candidates_per_prompt}",
        f"- 每题候选顺序独立随机，种子：{seed}",
        f"- 完整题集：{str(blind['complete_prompt_set']).lower()}",
        "",
        "> 评分期间不要打开 `quality_review_mapping.json` 或 `quality_generation.json`：",
        "> 前者直接给出标签到模型的映射，后者按模型标识保存回答，打开就会破坏盲评。",
        "",
        "## 评分维度（每维 0/1/2）",
        "",
        "| 维度 | 0 分 | 1 分 | 2 分 |",
        "| --- | --- | --- | --- |",
    ]

    for dimension in dimensions:
        anchors = payload["rubric"][dimension]
        label = DIMENSION_LABELS.get(dimension, dimension)
        lines.append(
            f"| {label} | {table_cell(anchors['0'])} | {table_cell(anchors['1'])} | {table_cell(anchors['2'])} |"
        )

    lines += [
        "",
        "## 填写方式",
        "",
        "每个候选回答下面都有一行待填，把四个下划线换成 0/1/2，并在 `|` 后写一句理由：",
        "",
        "```text",
        "A: 2 1 2 2 | 日期和地点正确，漏了报名截止日；单句无标题；无编造内容。",
        "```",
        "",
        f"- 四个数字依次是：{order}。",
        "- 只能填 0、1、2，理由不能为空。",
        "- 参考答案只是示例，摘要/改写/推理接受等价表述；缺信息记完整性，说错记正确性。",
        "- 同一题的三个候选各自独立打分，不要只排序。",
        "",
        "填完在项目根目录运行：",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe import_quality_scores.py",
        "```",
        "",
        "该命令只读本文件并写入 `quality_review_scores.json`，不会读取映射文件。",
        "",
        "---",
    ]

    for index, answer in enumerate(answers, start=1):
        category = CATEGORY_LABELS.get(answer["category"], answer["category"])

        lines += [
            "",
            f"## {index}. `{answer['prompt_id']}`（{category}）",
            "",
            "**Instruction**",
            "",
            fenced(answer["instruction"]),
            "",
            "**Input**",
            "",
            fenced(answer["input"]),
            "",
            "**参考答案（示例，不是唯一正确答案）**",
            "",
            fenced(answer["reference_answer"]),
            "",
            f"**必需信息**：{'；'.join(answer['required_facts'])}",
            "",
            f"**格式要求**：{answer['format_requirement']}",
            "",
            f"**禁止添加**：{'；'.join(answer['prohibited_additions'])}",
            "",
        ]

        for candidate in answer["candidates"]:
            lines += [
                f"### 候选 {candidate['label']}",
                "",
                fenced(candidate["response"]),
                "",
                f"{candidate['label']}: _ _ _ _ |",
                "",
            ]

    return "\n".join(lines) + "\n"


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)

    payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    validate_prompt_set(payload)

    output_dir = args.output_dir.resolve()
    written = []

    if args.from_report is not None:
        protect_outputs(output_dir, REVIEW_FILES, args.overwrite)

        report = json.loads(args.from_report.read_text(encoding="utf-8"))

        if report.get("prompt_set_sha256") != canonical_hash(payload):
            raise ValueError("generation report does not match the prompt-set hash")

        if report.get("evaluation_id") != payload["evaluation_id"]:
            raise ValueError("generation report has a different evaluation_id")

        print("re-exporting review files from", args.from_report)
    else:
        protect_outputs(output_dir, (GENERATION_REPORT, *REVIEW_FILES), args.overwrite)

        report = generate_report(payload, run_specs(args.adapter), args.limit)

        save_json(output_dir / GENERATION_REPORT, report)
        written.append(output_dir / GENERATION_REPORT)

    save_json(output_dir / BLIND_REVIEW, build_blind_review(payload, report, args.seed))
    save_json(output_dir / REVIEW_MAPPING, build_review_mapping(payload, report, args.seed))
    save_json(output_dir / SCORE_SHEET, build_score_sheet(payload, report))
    save_text(output_dir / WORKBOOK, build_review_workbook(payload, report, args.seed))

    written.extend(output_dir / name for name in REVIEW_FILES)

    print("prompts:", report["prompt_count"], "runs:", len(report_run_ids(report)))
    print("complete prompt set:", report["complete_prompt_set"])

    for path in written:
        print("saved:", path)

    print("Keep the review mapping separate from the scoring file until scoring is complete.")

    return written


if __name__ == "__main__":
    main()
