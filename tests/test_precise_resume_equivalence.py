import random
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from fate_oia.engine.precise_curriculum import curriculum_sha256, curriculum_state_for_epoch
from fate_oia.engine.train_precise_oia import load_resume_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class _TinyOwnedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.evidence_schema = [{"name": "field"}]


def _step(model, optimizer, scheduler):
    value = torch.randn(4, 3)
    scalar = random.random()
    loss = (model.linear(value).squeeze(-1) - scalar).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return value, scalar


def test_resume_restores_the_exact_next_optimizer_update(tmp_path):
    torch.manual_seed(20260722)
    random.seed(20260722)
    continuous = _TinyOwnedModel()
    optimizer_a = torch.optim.AdamW(continuous.parameters(), lr=1e-3)
    scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lambda step: 1.0)
    _step(continuous, optimizer_a, scheduler_a)
    fingerprint = {"source": "test"}
    curriculum_hash = "curriculum-test"
    owner_active_epochs = {"owner": 12}
    checkpoint = {
        "model": continuous.state_dict(), "optimizers": {"owner": optimizer_a.state_dict()},
        "schedulers": {"owner": scheduler_a.state_dict()}, "rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(), "cuda_rng_state": None, "epoch": 0,
        "global_optimizer_step": 1, "global_micro_step": 1, "optimizer_step_counts": {"owner": 1},
        "best_deploy_joint": 0.0, "best_scores": {}, "active_field_schema": ["field"],
        "implementation_fingerprint": fingerprint, "pcvl_optimizer_step_count": 0,
        "pcvl_nonzero_update_count": 0,
        "curriculum_sha256": curriculum_hash, "owner_active_epochs": owner_active_epochs,
        "curriculum_state": {"test": True}, "owner_step_deltas": {"owner": 1},
    }
    path = tmp_path / "resume.pth"
    torch.save(checkpoint, path)
    next_value, next_scalar = _step(continuous, optimizer_a, scheduler_a)

    resumed = _TinyOwnedModel()
    optimizer_b = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lambda step: 1.0)
    load_resume_checkpoint(
        path,
        resumed,
        {"owner": optimizer_b},
        {"owner": scheduler_b},
        torch.device("cpu"),
        fingerprint,
        curriculum_hash,
        owner_active_epochs,
    )
    resumed_value, resumed_scalar = _step(resumed, optimizer_b, scheduler_b)

    assert torch.equal(next_value, resumed_value)
    assert next_scalar == resumed_scalar
    for expected, actual in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("saved_epoch", [1, 3, 5])
def test_strict_resume_validates_curriculum_boundaries(saved_epoch, tmp_path):
    config = yaml.safe_load(
        (ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8")
    )
    owner = "reread_adapter"
    active_steps = sum(
        int(curriculum_state_for_epoch(config, epoch).owner_active[owner])
        for epoch in range(saved_epoch + 1)
    )
    model = _TinyOwnedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    for _ in range(active_steps):
        _step(model, optimizer, scheduler)
    state = curriculum_state_for_epoch(config, saved_epoch).to_dict()
    state.pop("owner_active")
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {owner: optimizer.state_dict()},
        "schedulers": {owner: scheduler.state_dict()},
        "rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": None,
        "epoch": saved_epoch,
        "global_optimizer_step": active_steps,
        "global_micro_step": active_steps,
        "optimizer_step_counts": {owner: active_steps},
        "best_deploy_joint": 0.0,
        "best_scores": {},
        "active_field_schema": ["field"],
        "implementation_fingerprint": {"source": "test"},
        "curriculum_sha256": curriculum_sha256(config),
        "owner_active_epochs": {owner: 10},
        "curriculum_state": state,
        "owner_step_deltas": {owner: int(curriculum_state_for_epoch(config, saved_epoch).owner_active[owner])},
    }
    path = tmp_path / f"resume_{saved_epoch}.pth"
    torch.save(checkpoint, path)
    resumed = _TinyOwnedModel()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 1.0)
    start_epoch, *_ = load_resume_checkpoint(
        path,
        resumed,
        {owner: resumed_optimizer},
        {owner: resumed_scheduler},
        torch.device("cpu"),
        {"source": "test"},
        curriculum_sha256(config),
        {owner: 10},
        config,
        {owner: 1},
    )
    assert start_epoch == saved_epoch + 1
    assert resumed_scheduler.last_epoch == active_steps
    assert bool(resumed_optimizer.state) == (active_steps > 0)


def test_strict_resume_fails_closed_when_step_counters_are_missing(tmp_path):
    model = _TinyOwnedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {"owner": optimizer.state_dict()},
        "schedulers": {"owner": scheduler.state_dict()},
        "rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "epoch": 0,
        "active_field_schema": ["field"],
        "implementation_fingerprint": {"source": "test"},
        "curriculum_sha256": "hash",
        "owner_active_epochs": {"owner": 1},
        "curriculum_state": {},
        "owner_step_deltas": {},
    }
    path = tmp_path / "missing.pth"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="optimizer_step_counts"):
        load_resume_checkpoint(
            path,
            _TinyOwnedModel(),
            {"owner": torch.optim.AdamW(_TinyOwnedModel().parameters(), lr=1e-3)},
            {"owner": scheduler},
            torch.device("cpu"),
            {"source": "test"},
            "hash",
            {"owner": 1},
        )
