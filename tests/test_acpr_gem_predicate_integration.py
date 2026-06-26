import torch

from fate_oia.models.acpr_scene_predicate_head import ACPRScenePredicateHead


def test_predicate_head_zero_init_evidence_is_equivalent_and_exposes_attention():
    head = ACPRScenePredicateHead("configs/acpr_scene_predicates.yaml", dim=64, num_layers=3)
    patch = torch.randn(2, 3, 32, 64)
    evidence = torch.randn(2, 20, 64)

    base = head(patch)
    gem = head(patch, evidence_tokens=evidence)

    assert torch.allclose(base["predicate_logits"], gem["predicate_logits"], atol=1e-6)
    assert gem["predicate_evidence_attention"].shape == (2, head.num_predicates, 20)
    loss = gem["predicate_logits"].sum()
    loss.backward()
    assert head.evidence_out_proj.weight.grad is not None
    assert head.evidence_out_proj.weight.grad.abs().sum() > 0
