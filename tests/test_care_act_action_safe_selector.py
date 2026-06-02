from __future__ import annotations

import torch

from fate_oia.engine.eval_care_act_oia import select_guarded_action_branch


def test_metric_level_selector_chooses_best_branch():
    labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    bad = torch.full_like(labels, -3.0)
    good = torch.where(labels > 0.5, torch.full_like(labels, 3.0), torch.full_like(labels, -3.0))
    candidates = {"base": bad, "evidence": good, "action_set": bad + 0.1, "candidate": bad - 0.1}
    selected = select_guarded_action_branch(candidates, labels, threshold=0.5, margin=0.006)
    assert selected["selected_branch"] == "evidence"
    assert torch.allclose(selected["guarded_logits"], good)


def test_selector_sets_shutdown_when_candidate_worse_than_base():
    labels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    base = torch.where(labels > 0.5, torch.full_like(labels, 3.0), torch.full_like(labels, -3.0))
    candidate = -base
    out = select_guarded_action_branch({"base": base, "candidate": candidate, "evidence": candidate, "action_set": candidate}, labels)
    assert out["selected_branch"] == "base"
    assert out["shutdown_action_residual_next_epoch"]
