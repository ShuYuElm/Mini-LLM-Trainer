import torch

def train_step(
    model,
    batch,
    optimizer,
    device,
    max_grad_norm=None,
    loss_divisor=1,
    should_update=True,
    scheduler=None,
    use_amp=False,
):
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }
    device_type = torch.device(device).type
    amp_enabled = use_amp and device_type == "cuda"

    with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
    ):
        outputs = model(**batch)
        loss = outputs.loss
        scaled_loss = loss / loss_divisor


    scaled_loss.backward()

    if should_update:

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return loss.item()

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    max_grad_norm=None,
    gradient_accumulation_steps=1,
    scheduler=None,
    use_amp=False,
):
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least 1"
        )

    num_updates = 0

    model.train()

    total_loss = 0
    num_steps = 0

    total_batches = len(dataloader)

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        group_start = (
            step // gradient_accumulation_steps
        ) * gradient_accumulation_steps

        current_group_size = min(
            gradient_accumulation_steps,
            total_batches - group_start,
        )

        should_update = (
            (step + 1) % gradient_accumulation_steps == 0
            or (step + 1) == total_batches
        )

        loss = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
            loss_divisor=current_group_size,
            should_update=should_update,
            scheduler=scheduler,
            use_amp=use_amp,
        )

        if should_update:
            optimizer.zero_grad()
            num_updates += 1

        total_loss += loss
        num_steps += 1

    avg_loss = total_loss / num_steps

    return avg_loss, num_updates