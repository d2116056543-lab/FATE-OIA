import torch

from fate_oia.models.evidence_token_builder import EvidenceTokenBuilder
from fate_oia.models.evidence_state_bottleneck import EvidenceStateBottleneck
from fate_oia.models.evis_oia_model import EviSOIAModel


def test_evidence_state_shapes_patch_only():
    tokens = torch.randn(2, 17, 64)
    builder = EvidenceTokenBuilder(64, evidence_mode="patch_only")
    evidence = builder(tokens, patch_grid=(4, 4))
    assert evidence.tokens.shape == (2, 0, 64)
    state = EvidenceStateBottleneck(64, num_state_queries=8, num_heads=4)
    out = state(tokens, evidence.tokens, evidence.mask)
    assert out["state_tokens"].shape == (2, 8, 64)
    assert out["state_attention"].shape[-1] == 17
    assert out["state_attention_entropy"].shape == (2,)


def test_evis_forward_outputs():
    model = EviSOIAModel(64, action_dim=4, reason_dim=21, num_state_queries=8, adaptive_calibration="global")
    labels = torch.randint(0, 2, (2, 21)).float()
    out = model(torch.randn(2, 33, 64), patch_grid=(4, 8), reason_labels=labels)
    assert out["action_logits_raw"].shape == (2, 4)
    assert out["reason_logits_raw"].shape == (2, 21)
    assert out["action_logits_calibrated"].shape == (2, 4)
    assert out["reason_logits_calibrated"].shape == (2, 21)
    assert out["mrc_reason_logits"].shape == (2, 21)
