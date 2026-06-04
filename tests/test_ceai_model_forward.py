import torch

from fate_oia.models.ceai_oia_model import CEAIOIAFeatureModel


def test_full_ceai_forward_required_keys_without_bdd100k_gt():
    model = CEAIOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, scene_proto_count=3, implicit_proto_count=2, pair_topk=4, expert_heads=4)
    tokens = torch.randn(2, 17, 32)
    out = model(tokens)
    required = [
        "base_action_logits", "base_reason_logits", "action_visual_logits", "action_reason_logits",
        "action_fused_logits", "reason_logits", "scene_state_logits", "implicit_prototypes",
        "action_specialist_logits", "reason_specialist_logits", "pair_support", "pair_reliability",
        "reason_reliability", "final_action_logits", "final_reason_logits", "router_action_gate",
        "router_reason_gate", "diagnostics",
    ]
    for key in required:
        assert key in out
    assert out["pair_support"].shape == (2, 4, 21)
    assert out["pair_reliability"].shape == (2, 4, 21)
