import torch

from fate_oia.models.acpr_scene_predicate_head import ACPRScenePredicateHead


def test_acpr_scene_predicate_head_outputs():
    head = ACPRScenePredicateHead("configs/acpr_scene_predicates.yaml")
    out = head(torch.randn(2, 3, 3600, 384))
    assert out["predicate_logits"].shape[0] == 2
    assert out["predicate_logits"].shape[1] >= 32
    assert out["predicate_attention"].shape[-1] == 3600
