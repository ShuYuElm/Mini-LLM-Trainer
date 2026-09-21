"""Inspect saved Alpaca splits without loading a model or starting training."""

from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from prepare_data import DATA_DIR
from src.dataset import SFTDataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        cache_dir=Path(__file__).resolve().parent / "model",
        local_files_only=True,
    )
    splits = load_from_disk(str(DATA_DIR))

    for split_name, raw_dataset in splits.items():
        lengths = []
        first_answer_tokens = []
        empty_answers = 0
        reference = SFTDataset(raw_dataset, tokenizer, max_length=128)

        for index, sample in enumerate(raw_dataset):
            instruction = sample["instruction"].strip()
            input_text = sample["input"].strip()
            answer = sample["output"].strip()
            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                prompt += f"### Input:\n{input_text}\n\n"
            prompt += "### Response:\n"
            text = prompt + answer + "\n\n"
            if tokenizer.eos_token is not None:
                text += tokenizer.eos_token

            encoded = tokenizer(
                text,
                truncation=False,
                padding=False,
                return_offsets_mapping=True,
            )
            lengths.append(len(encoded["input_ids"]))

            # Check representative rows against the actual Dataset formatter.
            if index in (0, len(raw_dataset) // 2, len(raw_dataset) - 1):
                assert encoded["input_ids"][:128] == reference[index]["input_ids"].tolist()

            if not answer:
                empty_answers += 1
                continue

            answer_start = len(prompt)
            answer_end = answer_start + len(answer)
            # Use offsets in the full text: separately tokenizing the prompt
            # can change the token boundary at the beginning of the answer.
            first_answer_tokens.append(next(
                token_index
                for token_index, (start, end) in enumerate(encoded["offset_mapping"])
                if end > answer_start and start < answer_end
            ))

        lengths = np.asarray(lengths)
        first_answer_tokens = np.asarray(first_answer_tokens)
        print(f"\n{split_name}: {len(lengths)} samples; empty answers: {empty_answers}")
        print("token length percentiles:", {
            f"p{p}": round(float(np.percentile(lengths, p)), 1)
            for p in (50, 90, 95, 99)
        })
        print("maximum token length:", int(lengths.max()))
        print("limit | truncated samples | nonempty answers fully lost | retained tokens")
        for limit in (128, 256, 512, 1024):
            truncated = int((lengths > limit).sum())
            lost_answers = int((first_answer_tokens >= limit).sum())
            retained = np.minimum(lengths, limit).sum() / lengths.sum()
            print(
                f"{limit:4} | {truncated:4}/{len(lengths)} ({truncated / len(lengths):.1%})"
                f" | {lost_answers:3}/{len(first_answer_tokens)}"
                f" | {retained:.1%}"
            )


if __name__ == "__main__":
    main()
