import torch

from fate_oia.models.pact_context_decoder import PACTContextDecoder
from fate_oia.models.pact_explanation_decoder import PACTExplanationDecoder
from fate_oia.models.pact_shared_readout import licensed_gradient


def _grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def test_action_and_reason_decoders_have_strict_owner_firewall():
    context, explanation = PACTContextDecoder(16), PACTExplanationDecoder(16)
    nodes = torch.randn(2, 25, 16, requires_grad=True)
    predicate = torch.randn(2, 5, 16)
    action = context(nodes, predicate)["action_logits_primary"].sum()
    action.backward(retain_graph=True)
    assert _grad_sum(context) > 0 and _grad_sum(explanation) == 0
    context.zero_grad(set_to_none=True)
    nodes.grad = None
    reason = explanation(licensed_gradient(nodes, 0.0), predicate)["reason_logits_visual_formal"].sum()
    reason.backward()
    assert _grad_sum(explanation) > 0 and _grad_sum(context) == 0
    assert float(nodes.grad.abs().max()) == 0.0


def test_license_scales_shared_gradient_linearly():
    base = torch.randn(2, 25, 8)
    gradients = []
    for license_value in (0.0, 0.5, 1.0):
        value = base.clone().requires_grad_()
        licensed_gradient(value, license_value).square().sum().backward()
        gradients.append(value.grad)
    assert float(gradients[0].abs().max()) == 0.0
    torch.testing.assert_close(gradients[1], gradients[2] * 0.5)
