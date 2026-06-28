from __future__ import annotations

import torch


def test_imports_and_basic_forward():
    from fate_oia.models.acpr_pmcal_v2_model import ACPRPMCalV2Model
    model = ACPRPMCalV2Model(
        use_mock_dino=True,
        scene_config="configs/acpr_scene_predicates.yaml",
        grammar_path="configs/acpr_reason_predicate_grammar.yaml",
        text_prompt_config="configs/acpr_pmcal_v2_text_prompts.yaml",
    )
    images = torch.randn(2, 3, 360, 640)
    action = torch.zeros(2, 4)
    reason = torch.zeros(2, 21)
    reason[:, 0] = 1
    out = model(images, split="train", action_labels=action, reason_labels=reason, file_names=["a.jpg", "b.jpg"])
    assert out["action_logits_base"].shape == (2, 4)
    assert out["reason_logits_base"].shape == (2, 21)
    assert out["q_pred"].shape[0] == 2
    assert out["predicate_attention"].shape[-1] == 3600
    assert torch.allclose(out["logits_deploy"], out["logits_base"] - out["threshold_logit"].view(1, -1), atol=1e-6)
