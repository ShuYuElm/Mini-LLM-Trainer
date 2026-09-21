import torch
import math

def evaluation_loss(model, dataloader, device, use_amp=False):
    was_training = model.training
    device_type = torch.device(device).type
    amp_enabled = use_amp and device_type == 'cuda'

    total_loss_sum = 0.0
    total_valid_tokens = 0

    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                num_valid_tokens = (
                    batch["labels"][:, 1:] != -100
                ).sum().item()

                if num_valid_tokens == 0:
                    continue

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    output = model(**batch)
                    loss = output.loss

                    total_loss_sum += loss.item() * num_valid_tokens
                    total_valid_tokens += num_valid_tokens

            if total_valid_tokens == 0:
                raise ValueError("No valid prediction tokens in evaluation data")

            return total_loss_sum / total_valid_tokens
    finally:
        model.train(was_training)

def compute_perplexity(loss):
    perplexity = math.exp(loss)
    return perplexity

def generate_response(
        model,
        tokenizer,
        instruction,
        device,
        input_text="",
        max_input_length=256,
        max_new_tokens=64,
        use_amp=False,
):
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=False,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    was_training = model.training
    device_type = torch.device(device).type
    amp_enabled = use_amp and device_type == 'cuda'

    model.eval()
    try:
        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                generated_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        prompt_length = encoded['input_ids'].shape[1]

        response_ids = generated_ids[
            0,
            prompt_length:,
        ]

        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()

        return response

    finally:
        model.train(was_training)