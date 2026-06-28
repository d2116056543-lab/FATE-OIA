from __future__ import annotations

import torch


def test_pmcal_final_action_independent_of_reason_labels():
    from fate_oia.models.acpr_pmcal_v2_model import ACPRPMCalV2Model
    model = ACPRPMCalV2Model(use_mock_dino=True)
    images = torch.randn(2, 3, 360, 640)
    action = torch.zeros(2, 4)
    reason_a = torch.zeros(2, 21)
    reason_b = torch.ones(2, 21)
    out_a = model(images, split="train", action_labels=action, reason_labels=reason_a, file_names=["a", "b"])
    out_b = model(images, split="train", action_labels=action, reason_labels=reason_b, file_names=["a", "b"])
    assert torch.allclose(out_a["action_logits_base"], out_b["action_logits_base"], atol=1e-6)
