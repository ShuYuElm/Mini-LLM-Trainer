"""Validate and fingerprint quality prompts; audit local lexical overlap only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parent
CATEGORIES = {"summarization", "rewriting", "extraction", "format_following", "reasoning"}
DIMENSIONS = {"correctness", "completeness", "format_compliance", "no_unsupported_additions"}


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_prompt_set(payload):
    require(isinstance(payload, dict) and payload.get("schema_version") == 1, "Expected schema_version 1 object")
    require(payload.get("evaluation_id") == "quality_eval_v1", "Unexpected evaluation_id")
    for key in ("language", "provenance", "purpose"):
        require(nonempty(payload.get(key)), f"Missing {key}")
    require(payload.get("generation_plan") == {"max_input_length": 256, "max_new_tokens": 128, "do_sample": False}, "Unexpected generation plan")
    rubric = payload.get("rubric")
    require(isinstance(rubric, dict) and set(rubric) == DIMENSIONS, "Expected four scoring dimensions")
    for name, anchors in rubric.items():
        require(isinstance(anchors, dict) and set(anchors) == {"0", "1", "2"}, f"{name}: scores must be 0/1/2")
        require(all(nonempty(v) for v in anchors.values()), f"{name}: empty scoring anchor")
    protocol = payload.get("review_protocol")
    require(isinstance(protocol, list) and protocol and all(nonempty(v) for v in protocol), "Missing review protocol")
    prompts = payload.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 40, "Expected 40 prompts")
    ids, signatures, categories = set(), set(), Counter()
    for prompt in prompts:
        require(isinstance(prompt, dict), "Prompt must be an object")
        for key in ("id", "category", "instruction", "reference_answer", "format_requirement"):
            require(nonempty(prompt.get(key)), f"Prompt missing {key}")
        name = prompt["id"]
        require(name not in ids, f"Duplicate ID: {name}")
        ids.add(name)
        require(prompt["category"] in CATEGORIES, f"{name}: unknown category")
        categories[prompt["category"]] += 1
        require(isinstance(prompt.get("input"), str), f"{name}: input must be a string")
        signature = normalize(prompt["instruction"] + "\n" + prompt["input"])
        require(signature not in signatures, f"{name}: duplicate normalized prompt")
        signatures.add(signature)
        for key in ("required_facts", "prohibited_additions"):
            values = prompt.get(key)
            require(isinstance(values, list) and values and all(nonempty(v) for v in values), f"{name}: missing {key}")
        check = prompt.get("automated_check")
        require(isinstance(check, dict), f"{name}: missing automated_check")
        kind = check.get("type")
        require(kind in {"human_only", "json_object", "exact_lines"}, f"{name}: unknown check type")
        if kind == "json_object":
            require(isinstance(check.get("expected"), dict) and check["expected"], f"{name}: expected JSON object missing")
            try:
                reference = json.loads(prompt["reference_answer"])
            except json.JSONDecodeError as error:
                raise ValueError(f"{name}: reference is not JSON") from error
            require(canonical_hash(reference) == canonical_hash(check["expected"]), f"{name}: JSON reference differs from expected values/types")
        elif kind == "exact_lines":
            expected = check.get("expected")
            require(isinstance(expected, list) and expected and all(nonempty(v) and "\n" not in v and "\r" not in v for v in expected), f"{name}: invalid expected lines")
            require(prompt["reference_answer"].strip().splitlines() == expected, f"{name}: line reference mismatch")
    require(dict(categories) == dict.fromkeys(CATEGORIES, 8), "Expected eight prompts per category")


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(text):
    words = text.split()
    return set(zip(words, words[1:], words[2:]))


def find_overlaps(prompts, corpus, threshold=0.35, minimum_shared=8):
    """Compare full prompts and substantive inputs, not instructions alone.

    Near matches use word-trigram Jaccard >= 0.35 and >= 8 shared trigrams.
    Short inputs (< 40 normalized characters) are not compared separately.
    """
    candidates = []
    for row in corpus:
        combined = normalize(row["instruction"] + "\n" + row["input"])
        text = normalize(row["input"])
        fields = {"prompt": combined}
        if len(text) >= 40:
            fields["input"] = text
        candidates.append((row, {key: (value, shingles(value)) for key, value in fields.items()}))
    matches = []
    for prompt in prompts:
        fields = {"prompt": normalize(prompt["instruction"] + "\n" + prompt["input"])}
        text = normalize(prompt["input"])
        if len(text) >= 40:
            fields["input"] = text
        for field, value in fields.items():
            grams = shingles(value)
            for row, other_fields in candidates:
                if field not in other_fields:
                    continue
                other, other_grams = other_fields[field]
                intersection = len(grams & other_grams)
                union = len(grams | other_grams)
                score = intersection / union if union else 0.0
                exact = value == other
                if exact or (intersection >= minimum_shared and score >= threshold):
                    matches.append({
                        "prompt_id": prompt["id"], "source": row["source"],
                        "source_id": row["source_id"], "field": field,
                        "kind": "normalized_exact" if exact else "near_lexical",
                        "jaccard": round(score, 6), "shared_trigrams": intersection,
                    })
    return matches


def check_prompt_lengths(prompts, tokenizer, limit):
    lengths = {}
    for prompt in prompts:
        text = f"### Instruction:\n{prompt['instruction'].strip()}\n\n"
        if prompt["input"].strip():
            text += f"### Input:\n{prompt['input'].strip()}\n\n"
        text += "### Response:\n"
        # Reference answers and criteria must never be passed to the tokenizer.
        length = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        require(length <= limit, f"{prompt['id']}: {length} tokens exceeds input limit {limit}")
        lengths[prompt["id"]] = length
    return lengths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/quality_eval_v1.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alpaca_seed42_train2000_val200")
    parser.add_argument("--development-prompts", type=Path, default=ROOT / "configs/eval_prompts.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/quality_eval_v1_audit.json")
    parser.add_argument("--check-tokenizer", action="store_true", help="Check input lengths using the cached Qwen tokenizer; no model loading.")
    args = parser.parse_args(argv)
    payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    validate_prompt_set(payload)
    development = json.loads(args.development_prompts.read_text(encoding="utf-8"))
    from datasets import load_from_disk
    splits = load_from_disk(str(args.data_dir))
    corpus, fingerprints = [], {}
    for split, size in (("train", 2000), ("validation", 200)):
        rows = list(splits[split])
        require(len(rows) == size, f"Expected {size} rows in {split}")
        fingerprints[split] = canonical_hash(rows)
        for row in rows:
            corpus.append({"source": split, "source_id": row["sample_id"], "instruction": row["instruction"], "input": row["input"]})
    for row in development:
        corpus.append({"source": "development", "source_id": row["id"], "instruction": row["instruction"], "input": row["input"]})
    lengths = None
    if args.check_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", cache_dir=ROOT / "model", local_files_only=True)
        lengths = check_prompt_lengths(payload["prompts"], tokenizer, payload["generation_plan"]["max_input_length"])
    matches = find_overlaps(payload["prompts"], corpus)
    report = {
        "evaluation_id": payload["evaluation_id"], "prompt_set_sha256": canonical_hash(payload),
        "hash_method": "SHA-256 of sorted-key UTF-8 canonical JSON; independent of file newline style",
        "category_counts": dict(Counter(p["category"] for p in payload["prompts"])),
        "sources": {"dataset_name": "yahma/alpaca-cleaned", "local_split": args.data_dir.name,
                    "counts": dict(Counter(row["source"] for row in corpus)), "split_sha256": fingerprints,
                    "development_prompts_sha256": canonical_hash(development)},
        "overlap_method": {"normalization": "Unicode NFKC, casefold, word characters; punctuation and whitespace ignored",
                           "fields": ["instruction + input", "input when >= 40 normalized characters"],
                           "near_match": "word-trigram Jaccard >= 0.35 AND >= 8 shared trigrams"},
        "matches": matches, "status": "needs_review" if matches else "no_lexical_matches",
        "input_token_lengths": lengths,
        "limitations": ["Lexical heuristics do not exclude semantic overlap or paraphrases.",
                        "Only the selected 2200 Alpaca rows and local development prompts were checked.",
                        "Unknown pretraining data and the remainder of Alpaca were not checked.",
                        "References were authored with prompts and require human review; no target-model outputs have been scored."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited 40 prompts against {len(corpus)} records: {len(matches)} lexical matches.")
    if lengths:
        print(f"Input tokens: {min(lengths.values())}..{max(lengths.values())}; limit 256.")
    print(f"Prompt-set SHA-256: {report['prompt_set_sha256']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
