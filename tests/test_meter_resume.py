from pathlib import Path

import torch
from torch import nn

from fate_oia.utils.meter_artifacts import load_checkpoint, save_checkpoint


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
