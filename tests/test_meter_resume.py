from pathlib import Path

import torch
from torch import nn

from fate_oia.utils.meter_artifacts import load_checkpoint, save_checkpoint
from vision_transformer import Attention


def test_checkpoint_restores_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=1, micro_step=2, optimizer_step=3, runtime_profile={}, meta_state={}, pu_state={}, calibration={}, config_hash="c", source_hash="s", schema_hash="h")
    restored = nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    payload = load_checkpoint(path, model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler, expected_config_hash="c", expected_source_hash="s", expected_schema_hash="h")
    assert payload["epoch"] == 1 and payload["optimizer_step"] == 3
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right)


def test_resume_next_update_matches_continuous_training(tmp_path: Path) -> None:
    torch.manual_seed(29)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    path = tmp_path / "next_update.pth"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        micro_step=0,
        optimizer_step=0,
        runtime_profile={},
        meta_state={},
        pu_state={},
        calibration={},
        config_hash="c",
        source_hash="implementation-head",
        schema_hash="h",
    )
    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 0.9**step)
    load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config_hash="c",
        expected_source_hash="implementation-head",
        expected_schema_hash="h",
    )
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    def update(candidate: nn.Module, candidate_optimizer, candidate_scheduler) -> torch.Tensor:
        candidate_optimizer.zero_grad(set_to_none=True)
        loss = (candidate(inputs) - targets).square().mean()
        loss.backward()
        candidate_optimizer.step()
        candidate_scheduler.step()
        return loss.detach()

    continuous_loss = update(model, optimizer, scheduler)
    resumed_loss = update(restored, restored_optimizer, restored_scheduler)
    torch.testing.assert_close(continuous_loss, resumed_loss)
    for continuous, resumed in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(continuous, resumed)


def test_dino_attention_projection_reference_is_not_registered_twice() -> None:
    attention = Attention(dim=8, num_heads=2)

    attention(torch.randn(2, 5, 8))

    assert attention.vproj is attention.proj
    assert not any(key.startswith("vproj.") for key in attention.state_dict())
