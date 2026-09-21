from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.trainer import train_one_epoch, train_step


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.last_batch_devices = None

    def forward(self, input_ids, attention_mask, labels):
        self.last_batch_devices = {
            "input_ids": input_ids.device,
            "attention_mask": attention_mask.device,
            "labels": labels.device,
        }

        predictions = self.weight * input_ids.float()
        token_losses = (predictions - labels.float()).pow(2)
        loss = (token_losses * attention_mask).sum() / attention_mask.sum()

        return SimpleNamespace(loss=loss)


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def make_batch():
    return {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "labels": torch.tensor([[2, 4]], dtype=torch.long),
    }


def test_train_step_returns_float_and_updates_parameter():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    returned_loss = train_step(
        model=model,
        batch=make_batch(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert isinstance(returned_loss, float)
    assert returned_loss == pytest.approx(2.5)
    assert model.weight.item() == pytest.approx(1.5)


def test_train_step_accumulates_into_existing_gradient():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.tensor(100.0)

    train_step(
        model=model,
        batch=make_batch(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert model.weight.grad.item() == pytest.approx(95.0)
    assert model.weight.item() == pytest.approx(-8.5)


def test_train_step_works_across_multiple_steps():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = make_batch()

    optimizer.zero_grad(set_to_none=True)
    first_loss = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    optimizer.zero_grad(set_to_none=True)
    second_loss = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert first_loss == pytest.approx(2.5)
    assert second_loss == pytest.approx(0.625)
    assert model.weight.item() == pytest.approx(1.75)


def test_train_step_passes_batch_on_requested_device():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    train_step(
        model=model,
        batch=make_batch(),
        optimizer=optimizer,
        device=device,
    )

    assert model.last_batch_devices == {
        "input_ids": device,
        "attention_mask": device,
        "labels": device,
    }


def test_train_step_clips_gradient_before_parameter_update():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    returned_loss = train_step(
        model=model,
        batch=make_batch(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
    )

    assert returned_loss == pytest.approx(2.5)
    assert model.weight.grad.item() == pytest.approx(-1.0)
    assert model.weight.item() == pytest.approx(1.1)


def test_train_one_epoch_updates_each_batch_and_returns_loss_and_update_count():
    model = TinyModel()
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [make_batch(), make_batch()]

    average_loss, num_updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert model.training is True
    assert isinstance(average_loss, float)
    assert average_loss == pytest.approx(1.5625)
    assert isinstance(num_updates, int)
    assert num_updates == 2
    assert model.weight.item() == pytest.approx(1.75)


def test_train_one_epoch_passes_gradient_clipping_to_each_step():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [make_batch(), make_batch()]

    average_loss, num_updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
    )

    assert average_loss == pytest.approx(2.2625)
    assert num_updates == 2
    assert model.weight.item() == pytest.approx(1.2)


def test_train_one_epoch_accumulates_two_micro_batches_before_update():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [make_batch(), make_batch()]

    average_loss, num_updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    assert average_loss == pytest.approx(2.5)
    assert num_updates == 1
    assert model.weight.item() == pytest.approx(1.5)
    assert model.weight.grad is None


def test_train_one_epoch_scales_incomplete_final_accumulation_group():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = [make_batch(), make_batch(), make_batch()]

    average_loss, num_updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    assert average_loss == pytest.approx(1.875)
    assert num_updates == 2
    assert model.weight.item() == pytest.approx(1.75)
    assert model.weight.grad is None


def test_train_one_epoch_rejects_invalid_accumulation_steps():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError):
        train_one_epoch(
            model=model,
            dataloader=[make_batch()],
            optimizer=optimizer,
            device=torch.device("cpu"),
            gradient_accumulation_steps=0,
        )


def test_train_one_epoch_steps_scheduler_once_per_optimizer_update():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    dataloader = [make_batch() for _ in range(5)]

    _, num_updates = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_accumulation_steps=2,
        scheduler=scheduler,
    )

    assert num_updates == 3
    assert scheduler.step_calls == num_updates


@pytest.mark.parametrize(
    ("batch_count", "accumulation_steps", "expected_updates"),
    [(4, 1, 4), (4, 2, 2), (5, 2, 3), (2, 4, 1)],
)
def test_reported_updates_match_actual_optimizer_steps(
    monkeypatch, batch_count, accumulation_steps, expected_updates
):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_step = Mock(wraps=optimizer.step)
    monkeypatch.setattr(optimizer, "step", optimizer_step)
    scheduler = CountingScheduler()

    _, num_updates = train_one_epoch(
        model=model,
        dataloader=[make_batch() for _ in range(batch_count)],
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_accumulation_steps=accumulation_steps,
        scheduler=scheduler,
    )

    assert num_updates == expected_updates
    assert optimizer_step.call_count == num_updates
    assert scheduler.step_calls == num_updates


def test_epoch_update_counts_can_accumulate_into_global_step():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    dataloader = [make_batch() for _ in range(5)]
    global_step = 0
    progress = []

    for _ in range(2):
        _, epoch_updates = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            gradient_accumulation_steps=2,
            scheduler=scheduler,
        )
        # Each epoch reports its own updates, not the running total.
        assert epoch_updates == 3
        global_step += epoch_updates
        progress.append(global_step)

    assert progress == [3, 6]
    assert scheduler.step_calls == global_step


def test_train_one_epoch_disables_amp_on_cpu(monkeypatch):
    autocast_calls = []
    original_autocast = torch.autocast

    def recording_autocast(*, device_type, dtype, enabled):
        autocast_calls.append(
            {
                "device_type": device_type,
                "dtype": dtype,
                "enabled": enabled,
            }
        )
        return original_autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        )

    monkeypatch.setattr(torch, "autocast", recording_autocast)

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(
        model=model,
        dataloader=[make_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        use_amp=True,
    )

    assert autocast_calls == [
        {
            "device_type": "cpu",
            "dtype": torch.bfloat16,
            "enabled": False,
        }
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to verify enabled AMP",
)
def test_train_step_enables_bfloat16_amp_on_cuda(monkeypatch):
    autocast_calls = []
    original_autocast = torch.autocast

    def recording_autocast(*, device_type, dtype, enabled):
        autocast_calls.append(
            {
                "device_type": device_type,
                "dtype": dtype,
                "enabled": enabled,
            }
        )
        return original_autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        )

    monkeypatch.setattr(torch, "autocast", recording_autocast)

    device = torch.device("cuda")
    model = TinyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_step(
        model=model,
        batch=make_batch(),
        optimizer=optimizer,
        device=device,
        use_amp=True,
    )

    assert autocast_calls == [
        {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": True,
        }
    ]
