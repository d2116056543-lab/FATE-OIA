import torch

from fate_oia.models.acpr_grounded_evidence_memory import ACPRGroundedEvidencePooler, load_evidence_slot_specs


def test_evidence_pooler_returns_named_sparse_slots():
    specs = load_evidence_slot_specs("configs/acpr_gem_evidence_slots.yaml")
    pooler = ACPRGroundedEvidencePooler(dim=384, slot_specs=specs, topk=32)
    tokens = torch.randn(2, 3, 3600, 384)

    out = pooler(tokens)

    assert out["evidence_tokens"].shape == (2, 20, 384)
    assert out["evidence_attention"].shape == (2, 20, 3600)
    assert len(out["evidence_slot_names"]) == 20
    assert set(out["evidence_slot_groups"]) >= {"object", "lane", "drivable", "traffic", "context"}
    assert torch.allclose(out["evidence_attention"].sum(-1), torch.ones(2, 20), atol=1e-4)
    assert (out["evidence_attention"] == 0).float().mean() > 0.5


def test_evidence_pooler_can_accept_grounding_targets():
    specs = load_evidence_slot_specs("configs/acpr_gem_evidence_slots.yaml")
    pooler = ACPRGroundedEvidencePooler(dim=384, slot_specs=specs, topk=16)
    tokens = torch.randn(1, 3, 3600, 384)
    targets = torch.zeros(1, 20, 3600)
    masks = torch.zeros(1, 20)
    targets[:, 0, :50] = 1
    masks[:, 0] = 1

    out = pooler(tokens, grounding_targets=targets, grounding_mask=masks)

    assert out["evidence_grounding_targets"].shape == (1, 20, 3600)
    assert out["evidence_grounding_mask"].shape == (1, 20)
    assert out["evidence_available_rate"].item() > 0
