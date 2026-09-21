import torch
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.evaluation import generate_response, evaluation_loss, compute_perplexity
from src.lora import inject_lora
from src.utils import load_lora_adapter, save_json
from src.dataset import SFTDataset
from src.collator import SFTDataCollator
from prepare_data import DATA_DIR

MAX_LENGTH = 512
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_CACHE_DIR = Path(__file__).resolve().parent / "model"
PROMPT_PATH = PROJECT_ROOT / "configs" / "eval_prompts.json"

MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 128


def load_adapter_for_evaluation(model, adapter_path, base_model_name):
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)

    if payload["base_model_name"] != base_model_name:
        raise ValueError("adapter base model does not match")

    model.requires_grad_(False)

    inject_lora(
        model=model,
        rank=payload["rank"],
        alpha=payload["alpha"],
        target_names=payload["target_names"]
    )

    metadata = load_lora_adapter(adapter_path, model)

    return metadata

def generate_samples(model, tokenizer, prompts, device):
    results = []

    for sample in prompts:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            instruction=sample["instruction"],
            input_text=sample["input"],
            device=device,
            max_input_length=MAX_INPUT_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS,
            use_amp=True
        )

        results.append({
            "id": sample["id"],
            "response": response
        })

        print("generated:", sample["id"])

    return results

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the base model with a LoRA adapter."
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to the saved LoRA adapter.",
    )

    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Compute loss only on the answer suffix.",
    )

    args = parser.parse_args(argv)

    if not args.adapter.is_file():
        parser.error(f"Adapter file does not exist: {args.adapter}")

    return args

def main():
    args = parse_args()

    adapter_path = args.adapter.resolve()
    answer_only = args.answer_only
    label_policy = "answer_only" if answer_only else "full_sequence"

    report_path = adapter_path.parent / (
        "evaluation_answer_only.json"
        if answer_only
        else "evaluation.json"
    )

    with PROMPT_PATH.open("r", encoding="utf-8") as file:
        prompts = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )

    splits = load_from_disk(str(DATA_DIR))
    raw_val_dataset = splits["validation"]

    assert len(raw_val_dataset) == 200

    print("validation samples:", len(raw_val_dataset))

    val_dataset = SFTDataset(
        raw_val_dataset,
        tokenizer,
        max_length=MAX_LENGTH,
        answer_only=answer_only
    )

    collator = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        dtype=torch.bfloat16,
    )

    model.to(device)

    base_loss = evaluation_loss(
        model=model,
        dataloader=val_dataloader,
        device=device,
        use_amp=True,
    )
    base_ppl = compute_perplexity(base_loss)

    print("Generating Base responses...")

    base_samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    metadata = load_adapter_for_evaluation(
        model=model,
        adapter_path=adapter_path,
        base_model_name=MODEL_NAME
    )

    lora_loss = evaluation_loss(
        model=model,
        dataloader=val_dataloader,
        device=device,
        use_amp=True,
    )
    lora_ppl = compute_perplexity(lora_loss)


    print("Generating LoRA responses...")

    lora_samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    report = {
        "model_name": MODEL_NAME,
        "adapter_path": str(adapter_path),
        "adapter_metadata": metadata,
        "data_dir": str(DATA_DIR.resolve()),
        "validation_samples": len(val_dataset),
        "validation_max_length": MAX_LENGTH,
        "label_policy": label_policy,
        "generation": {
            "max_input_length": MAX_INPUT_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False, 
        },
        "prompts": prompts,
        "base": {
            "loss": base_loss,
            "perplexity": base_ppl,
            "samples": base_samples,
        },
        "lora": {
            "loss": lora_loss,
            "perplexity": lora_ppl,
            "samples": lora_samples
        }
    }

    save_json(report_path, report)

    print(f"Base: loss={base_loss:.4f}, perplexity={base_ppl:.4f}")
    print(f"LoRA: loss={lora_loss:.4f}, perplexity={lora_ppl:.4f}")
    print("Base responses:", len(base_samples))
    print("LoRA responses:", len(lora_samples))
    print("Evaluation report saved:", report_path)


if __name__ == "__main__":
    main()