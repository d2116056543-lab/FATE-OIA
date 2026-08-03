import random
from pathlib import Path

import numpy as np
import pytest
import torch

from fate_oia.utils.save_artifacts import load_checkpoint, save_checkpoint


def _stateful_model():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path: Path):
    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    model, optimizer, scheduler = _stateful_model()
    (model(torch.ones(1, 2)).sum()).backward(); optimizer.step(); scheduler.step()
    path = save_checkpoint(
        tmp_path / "latest.pth", model=model, optimizer=optimizer, scheduler=scheduler,
        optimizer_step=1, action_rms_ema={"a": 1}, view_consistency_ema={"r": 1},
        utility_cadence={"phase": 1}, tail_prototypes={"p": 1}, pu_lambda={"r": .2},
        calibration={"theta": [0]}, split_manifest={"main": [1]}, git_head="a" * 40,
        config_hash="c", source_tree_hash="s", schema_hash="x", file_order_hash="o",
    )
    expected = torch.rand(3)
    torch.rand(5); random.random(); np.random.rand()
    restored, restored_optim, restored_sched = _stateful_model()
    payload = load_checkpoint(
        path, model=restored, optimizer=restored_optim, scheduler=restored_sched,
        expected_git_head="a" * 40, expected_config_hash="c", expected_source_tree_hash="s",
        expected_schema_hash="x", expected_file_order_hash="o",
    )
    assert payload["optimizer_step"] == 1
    assert torch.equal(torch.rand(3), expected)
    for expected_parameter, actual_parameter in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected_parameter, actual_parameter)


def test_checkpoint_fails_closed_on_hash_mismatch(tmp_path: Path):
    model, _, _ = _stateful_model()
    path = save_checkpoint(
        tmp_path / "latest.pth", model=model, optimizer_step=0, git_head="a" * 40,
        config_hash="c", source_tree_hash="s", schema_hash="x", file_order_hash="o",
    )
    with pytest.raises(ValueError, match="config_hash mismatch"):
        load_checkpoint(path, model=model, expected_config_hash="wrong")
