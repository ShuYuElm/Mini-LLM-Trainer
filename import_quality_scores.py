"""Import hand-written scores from the review workbook into the score sheet.

Reads the filled Markdown workbook and the empty score sheet, validates every
score line, and writes the machine-readable sheet. The label-to-run mapping is
never read, so blinding is preserved.
"""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from audit_quality_eval import canonical_hash


ROOT = Path(__file__).resolve().parent
REVIEW_DIR = ROOT / "experiments" / "quality_eval_v1"
DEFAULT_PROMPTS = ROOT / "configs" / "quality_eval_v1.json"
DEFAULT_WORKBOOK = REVIEW_DIR / "quality_review_workbook.md"
DEFAULT_SCORES = REVIEW_DIR / "quality_review_scores.json"

DIMENSIONS = ("correctness", "completeness", "format_compliance", "no_unsupported_additions")

HASH_MARKER = re.compile(r"<!--\s*prompt_set_sha256:\s*(?P<value>[0-9a-f]{64})\s*-->")
HEADING = re.compile(r"^##\s+\d+\.\s+`(?P<prompt_id>[^`]+)`", re.MULTILINE)
SCORE_LINE = re.compile(
    r"^(?P<label>[A-Z]):[ \t]*(?P<scores>[^|\n]*?)[ \t]*\|[ \t]*(?P<rationale>[^\n]*?)[ \t]*$",
    re.MULTILINE,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="Filled Markdown scoring workbook.",
    )

    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help="Score sheet to write; the same file is read to check prompt order.",
    )

    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Quality prompt set, used for the prompt-set hash.",
    )

    args = parser.parse_args(argv)

    for name in ("workbook", "scores", "prompts"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")

    return args


def parse_workbook(text):
    """Return the prompt-set hash and {(prompt_id, label): (score text, rationale)}."""
    marker = HASH_MARKER.search(text)

    if marker is None:
        raise ValueError("workbook has no prompt-set hash marker; regenerate it with generate_quality_eval.py")

    headings = list(HEADING.finditer(text))

    if not headings:
        raise ValueError("workbook has no prompt sections")

    entries = {}
    duplicates = []

    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        prompt_id = heading.group("prompt_id")

        for line in SCORE_LINE.finditer(text[start:end]):
            key = (prompt_id, line.group("label"))

            if key in entries:
                duplicates.append(f"{prompt_id} candidate {line.group('label')}: duplicate score line")
                continue

            entries[key] = (line.group("scores").strip(), line.group("rationale").strip())

    if duplicates:
        raise ValueError("; ".join(duplicates))

    return marker.group("value"), entries


def score_values(raw, prompt_id, label):
    if "_" in raw:
        raise ValueError(f"{prompt_id} candidate {label}: not scored yet")

    tokens = raw.split()

    if len(tokens) != len(DIMENSIONS):
        raise ValueError(
            f"{prompt_id} candidate {label}: expected {len(DIMENSIONS)} scores, found {len(tokens)}"
        )

    values = []

    for token in tokens:
        if token not in {"0", "1", "2"}:
            raise ValueError(f"{prompt_id} candidate {label}: '{token}' is not 0, 1, or 2")
        values.append(int(token))

    return values


def collect_problems(sheet, entries):
    """Return every reason the workbook cannot be imported yet."""
    problems = []
    expected = set()

    for entry in sheet["entries"]:
        key = (entry["prompt_id"], entry["label"])
        expected.add(key)
        name = f"{entry['prompt_id']} candidate {entry['label']}"

        if key not in entries:
            problems.append(f"{name}: no score line in the workbook")
            continue

        raw, rationale = entries[key]

        try:
            score_values(raw, entry["prompt_id"], entry["label"])
        except ValueError as error:
            problems.append(str(error))
            continue

        if not rationale:
            problems.append(f"{name}: empty rationale")

    for prompt_id, label in sorted(set(entries) - expected):
        problems.append(f"{prompt_id} candidate {label}: not part of this score sheet")

    return problems


def fill_sheet(sheet, entries):
    """Return a filled copy of the score sheet; raise ValueError listing problems."""
    if list(sheet["entries"][0]["scores"]) != list(DIMENSIONS):
        raise ValueError("score sheet dimensions differ from the workbook column order")

    problems = collect_problems(sheet, entries)

    if problems:
        raise ValueError("\n".join(problems))

    filled = copy.deepcopy(sheet)

    for entry in filled["entries"]:
        raw, rationale = entries[(entry["prompt_id"], entry["label"])]
        entry["scores"] = dict(zip(DIMENSIONS, score_values(raw, entry["prompt_id"], entry["label"])))
        entry["rationale"] = rationale

    return filled


def summarize(sheet):
    """Score distribution per dimension; no model identity is involved."""
    summary = {}

    for dimension in DIMENSIONS:
        tally = Counter(entry["scores"][dimension] for entry in sheet["entries"])
        summary[dimension] = {score: tally.get(score, 0) for score in (0, 1, 2)}

    return summary


def dump_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)


def main(argv=None):
    args = parse_args(argv)

    payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    sheet = json.loads(args.scores.read_text(encoding="utf-8"))

    try:
        workbook_hash, entries = parse_workbook(args.workbook.read_text(encoding="utf-8"))
    except ValueError as error:
        print("cannot read the workbook:", error)
        raise SystemExit(1) from error

    expected_hash = canonical_hash(payload)

    for name, value in (("workbook", workbook_hash), ("score sheet", sheet.get("prompt_set_sha256"))):
        if value != expected_hash:
            print(f"{name} does not match the prompt set: {value} != {expected_hash}")
            raise SystemExit(1)

    problems = collect_problems(sheet, entries)

    if problems:
        print(f"the workbook is not ready: {len(problems)} problems")
        for problem in problems[:10]:
            print(" -", problem)
        if len(problems) > 10:
            print(f" - ... and {len(problems) - 10} more")
        print("nothing written; fix the workbook and run again")
        raise SystemExit(1)

    already = sum(entry["scores"][DIMENSIONS[0]] is not None for entry in sheet["entries"])
    filled = fill_sheet(sheet, entries)

    args.scores.write_text(dump_json(filled), encoding="utf-8")

    prompts = {entry["prompt_id"] for entry in filled["entries"]}

    print(f"recorded {len(filled['entries'])} scores for {len(prompts)} prompts")

    if already:
        print(f"replaced {already} previously recorded scores")

    for dimension, tally in summarize(filled).items():
        print(f"  {dimension}: 0={tally[0]} 1={tally[1]} 2={tally[2]}")

    print("saved:", args.scores)
    print("next: reveal quality_review_mapping.json and aggregate the scores")


if __name__ == "__main__":
    main()
