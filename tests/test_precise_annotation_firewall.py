import torch

from fate_oia.models.precise_annotation_head import PRECISEAnnotationHead


def test_observed_annotation_loss_updates_only_annotation_adapter():
    head = PRECISEAnnotationHead()
    semantic_tokens = torch.randn(2, 21, 384, requires_grad=True)
    context = torch.randn(2, 384, requires_grad=True)
    semantic_logits = torch.randn(2, 21, requires_grad=True)
    output = head(semantic_tokens, context, semantic_logits)
    output["reason_logits_observed"].square().mean().backward()
    assert semantic_tokens.grad is None
    assert context.grad is None
    assert semantic_logits.grad is None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_annotation_delta_is_context_dependent_and_bounded():
    head = PRECISEAnnotationHead()
    token = torch.randn(2, 21, 384)
    semantic = torch.randn(2, 21)
    output = head(token, torch.randn(2, 384), semantic)
    assert output["annotation_delta"].abs().max().item() <= 0.750001
