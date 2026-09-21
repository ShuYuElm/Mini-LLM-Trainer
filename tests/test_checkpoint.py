import pytest
import torch

from src.utils import load_checkpoint, save_checkpoint


def make_training_objects():
    model = torch.nn.Linear(2, 1)
    # Persistent buffers belong in state_dict(), even though they are not parameters.
    model.register_buffer("example_buffer", torch.tensor([3.0]))
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    return model, optimizer, scheduler


@pytest.mark.parametrize("path_as_string", [False, True])
def test_save_checkpoint_creates_parent_directories_and_handles_no_scheduler(
    tmp_path, path_as_string
):
    model, optimizer, _ = make_training_objects()
    checkpoint_path = tmp_path / "nested" / "checkpoints" / "epoch_1.pt"
    path_argument = str(checkpoint_path) if path_as_string else checkpoint_path

    save_checkpoint(
        path=path_argument,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        global_step=2,
    )

    assert checkpoint_path.is_file()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    assert set(checkpoint) == {
        "model", "optimizer", "scheduler", "epoch", "global_step"
    }
    assert checkpoint["scheduler"] is None
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 2


def test_save_checkpoint_preserves_model_optimizer_scheduler_and_progress(tmp_path):
    model, optimizer, scheduler = make_training_objects()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # AdamW's moment estimates are created by step(), not by its constructor.
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    checkpoint_path = tmp_path / "epoch_1.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=2,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    expected_model_state = model.state_dict()
    assert checkpoint["model"].keys() == expected_model_state.keys()
    for name, tensor in expected_model_state.items():
        torch.testing.assert_close(
            checkpoint["model"][name], tensor, rtol=0, atol=0
        )

    expected_optimizer_state = optimizer.state_dict()
    saved_optimizer_state = checkpoint["optimizer"]
    assert expected_optimizer_state["state"]
    assert saved_optimizer_state.keys() == expected_optimizer_state.keys()
    assert (
        saved_optimizer_state["param_groups"]
        == expected_optimizer_state["param_groups"]
    )
    assert (
        saved_optimizer_state["state"].keys()
        == expected_optimizer_state["state"].keys()
    )
    for parameter_id, parameter_state in expected_optimizer_state["state"].items():
        saved_parameter_state = saved_optimizer_state["state"][parameter_id]
        assert saved_parameter_state.keys() == parameter_state.keys()
        for name, tensor in parameter_state.items():
            torch.testing.assert_close(
                saved_parameter_state[name], tensor, rtol=0, atol=0
            )

    assert checkpoint["scheduler"] == scheduler.state_dict()
    assert checkpoint["scheduler"]["last_epoch"] == 2
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 2


def test_saved_checkpoint_is_not_changed_by_later_model_updates(tmp_path):
    model, optimizer, scheduler = make_training_objects()
    original_weight = model.weight.detach().clone()
    checkpoint_path = tmp_path / "before_update.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    torch.testing.assert_close(
        checkpoint["model"]["weight"], original_weight, rtol=0, atol=0
    )
    assert not torch.equal(checkpoint["model"]["weight"], model.weight)


def run_toy_training_step(model, optimizer, scheduler):
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    optimizer.zero_grad()
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


@pytest.mark.parametrize("with_scheduler", [False, True])
def test_load_checkpoint_resumes_the_same_training_trajectory(tmp_path, with_scheduler):
    model, optimizer, scheduler = make_training_objects()
    if not with_scheduler:
        scheduler = None
    for _ in range(2):
        run_toy_training_step(model, optimizer, scheduler)
    model.example_buffer.fill_(7.0)

    checkpoint_path = tmp_path / "epoch_1.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=2,
    )

    # These are new objects, with fresh parameters and empty AdamW state.
    restored_model, restored_optimizer, restored_scheduler = make_training_objects()
    if not with_scheduler:
        restored_scheduler = None
    assert not restored_optimizer.state
    assert not torch.equal(restored_model.weight, model.weight)

    epoch, global_step = load_checkpoint(
        path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert epoch == 1
    assert global_step == 2
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            restored_model.state_dict()[name], tensor, rtol=0, atol=0
        )
    assert restored_optimizer.state
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    # Saving must not interrupt the original branch. Both branches take two
    # further updates with identical data and should end in exactly the same state.
    for _ in range(2):
        expected_loss = run_toy_training_step(model, optimizer, scheduler)
        restored_loss = run_toy_training_step(
            restored_model, restored_optimizer, restored_scheduler
        )
        assert restored_loss == expected_loss

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(
            restored_model.state_dict()[name], tensor, rtol=0, atol=0
        )

    expected_optimizer_state = optimizer.state_dict()
    restored_optimizer_state = restored_optimizer.state_dict()
    assert (
        restored_optimizer_state["param_groups"]
        == expected_optimizer_state["param_groups"]
    )
    assert (
        restored_optimizer_state["state"].keys()
        == expected_optimizer_state["state"].keys()
    )
    for parameter_id, state in expected_optimizer_state["state"].items():
        restored_state = restored_optimizer_state["state"][parameter_id]
        assert restored_state.keys() == state.keys()
        for name, tensor in state.items():
            torch.testing.assert_close(
                restored_state[name], tensor, rtol=0, atol=0
            )

    if with_scheduler:
        assert restored_scheduler.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize("saved_with_scheduler", [False, True])
def test_load_checkpoint_rejects_scheduler_mismatch_before_changing_state(
    tmp_path, saved_with_scheduler
):
    model, optimizer, scheduler = make_training_objects()
    if not saved_with_scheduler:
        scheduler = None
    run_toy_training_step(model, optimizer, scheduler)
    checkpoint_path = tmp_path / "epoch_1.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
    )

    restored_model, restored_optimizer, restored_scheduler = make_training_objects()
    if saved_with_scheduler:
        restored_scheduler = None
    original_state = {
        name: tensor.clone() for name, tensor in restored_model.state_dict().items()
    }

    with pytest.raises(ValueError, match="scheduler"):
        load_checkpoint(
            path=checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )

    for name, tensor in original_state.items():
        torch.testing.assert_close(
            restored_model.state_dict()[name], tensor, rtol=0, atol=0
        )
    assert not restored_optimizer.state
