import torch

from fate_oia.models.acpr_grounded_evidence_memory import ACPREvidenceOraclePooler, ACPRGroundedEvidencePooler, load_evidence_slot_specs


def test_oracle_pooler_uses_masks_and_reports_availability():
    specs = load_evidence_slot_specs("configs/acpr_gem_evidence_slots.yaml")
    learned = ACPRGroundedEvidencePooler(dim=384, slot_specs=specs, topk=16)
    oracle = ACPREvidenceOraclePooler(learned)
    patch = torch.randn(2, 3600, 384)
    masks = torch.zeros(2, 20, 3600)
    masks[:, 0, :4] = 1

    out = oracle(patch, masks)

    expected = patch[:, :4].mean(1)
    assert torch.allclose(out["evidence_tokens"][:, 0], expected, atol=1e-5)
    assert out["oracle_available"][:, 0].all()
    assert not out["oracle_available"][:, 1].any()
    assert out["evidence_oracle_mode"] is True
