from __future__ import annotations

import torch

from fate_oia.models.care_act_model import CAREActOIAModel


def test_care_act_forward_outputs_required_branches_and_caps():
    model = CAREActOIAModel(action_evidence_cap_max=0.12, action_set_cap=0.06)
    model.train()
    tokens = torch.randn(2, 45 * 80 + 1, 384)
    reason = torch.zeros(2, 21)
    reason[:, [5, 6, 9]] = 1
    out = model(tokens, batch={"reason": reason}, structured=[None, None], epoch=8)
    for key in [
        "action_base_logits",
        "action_visual_logits",
        "action_reason_logits",
        "reason_to_action_logits",
        "action_evidence_logits",
        "action_set_logits",
        "action_final_candidate_logits",
        "action_guarded_logits",
        "reason_base_logits",
        "reason_logits",
        "action_branch_candidates",
    ]:
        assert key in out
    assert out["action_evidence_delta"].abs().max().item() <= 0.120001
    assert out["action_set_delta"].abs().max().item() <= 0.060001
    assert out["action_total_delta"].abs().max().item() <= 0.150001
    assert out["diagnostics"]["primary_test_uses_bdd100k_gt"] is False


def test_primary_test_is_image_only_even_if_structured_supplied():
    model = CAREActOIAModel()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 45 * 80 + 1, 384), batch=None, structured=[{"objects": [{"category": "car"}]}], epoch=8)
    assert out["diagnostics"]["primary_test_uses_bdd100k_gt"] is False
