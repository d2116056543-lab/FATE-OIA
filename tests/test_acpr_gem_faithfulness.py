import torch

from fate_oia.engine.eval_acpr_gem_faithfulness import deletion_drop


def test_top_evidence_deletion_drop_exceeds_random_when_top_is_causal():
    logits = torch.tensor([[2.0, 1.0]])
    top_deleted = torch.tensor([[0.5, 1.0]])
    random_deleted = torch.tensor([[1.8, 1.0]])

    top = deletion_drop(logits, top_deleted)
    random = deletion_drop(logits, random_deleted)

    assert top > random
