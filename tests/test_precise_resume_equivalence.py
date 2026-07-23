import random

import torch
from torch import nn

from fate_oia.engine.train_precise_oia import load_resume_checkpoint


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
    checkpoint = {
        "model": continuous.state_dict(), "optimizers": {"owner": optimizer_a.state_dict()},
        "schedulers": {"owner": scheduler_a.state_dict()}, "rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(), "cuda_rng_state": None, "epoch": 0,
        "global_optimizer_step": 1, "global_micro_step": 1, "optimizer_step_counts": {"owner": 1},
        "best_deploy_joint": 0.0, "best_scores": {}, "active_field_schema": ["field"],
        "implementation_fingerprint": fingerprint, "pcvl_optimizer_step_count": 0,
        "pcvl_nonzero_update_count": 0,
    }
    path = tmp_path / "resume.pth"
    torch.save(checkpoint, path)
    next_value, next_scalar = _step(continuous, optimizer_a, scheduler_a)

    resumed = _TinyOwnedModel()
    optimizer_b = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lambda step: 1.0)
    load_resume_checkpoint(path, resumed, {"owner": optimizer_b}, {"owner": scheduler_b}, torch.device("cpu"), fingerprint)
    resumed_value, resumed_scalar = _step(resumed, optimizer_b, scheduler_b)

    assert torch.equal(next_value, resumed_value)
    assert next_scalar == resumed_scalar
    for expected, actual in zip(continuous.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)
