from types import SimpleNamespace
import torch
from fate_oia.losses.action_primary_trace_losses import compute_action_primary_trace_loss


def _out():
    b = 3
    ad = 4
    rd = 21
    return {
        "action_logits": torch.randn(b, ad, requires_grad=True),
        "action_logits_base_plus_bias": torch.randn(b, ad, requires_grad=True),
        "base_action_visual_logits": torch.randn(b, ad, requires_grad=True),
        "base_action_reason_logits": torch.randn(b, ad, requires_grad=True),
        "action_logits_reason_to_action": torch.randn(b, ad, requires_grad=True),
        "reason_logits": torch.randn(b, rd, requires_grad=True),
        "base_reason_logits": torch.randn(b, rd, requires_grad=True),
        "transport": {"evidence_reason_logits": torch.randn(b, rd, requires_grad=True), "transport_entropy": torch.rand(b, rd), "T_sparse_fraction": torch.tensor(0.8)},
        "action_bias_eff": torch.randn(ad, requires_grad=True),
    }


def test_action_primary_loss_has_split_components_and_gradients():
    args = SimpleNamespace(action_dim=4, asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05, action_asl=1.3, action_visual_aux=0.06, action_reason_aux=0.12, reason_to_action_gt=0.16, action_agreement=0.01, action_bias_l2=0.002, reason_asl=0.70, evidence_reason_asl=0.10, evidence_reason_rank=0.02, evidence_base_distill=0.01, prototype_diversity=0.0, transport_entropy=0.001, conflict_gate_enabled=False)
    labels = (torch.rand(3, 25) > 0.7).float()
    out = _out()
    loss, stats = compute_action_primary_trace_loss(args, out, labels, epoch=0)
    assert loss.requires_grad
    for key in ["loss_action_main", "loss_action_visual_aux", "loss_action_reason_aux", "loss_r2a_gt", "loss_reason_asl", "loss_evidence_reason_asl", "action_loss_total", "reason_loss_total", "evidence_loss_total"]:
        assert key in stats
    loss.backward()
    assert out["action_bias_eff"].grad is not None
