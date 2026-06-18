import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_seca_model_forward_contract():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, seca_enabled=True)
    out = model(torch.randn(1, 3, 360, 640))
    for key in [
        "action_logits_legacy_base",
        "action_nodes_legacy",
        "action_nodes_seca",
        "seca_action_reason_attention",
        "seca_null_attention",
        "reason_predicate_attention",
    ]:
        assert key in out
    assert out["seca_action_reason_attention"].shape == (1, 4, 22)
    assert out["reason_predicate_attention"].shape[2] == 21
