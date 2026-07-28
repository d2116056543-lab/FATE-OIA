import torch

from fate_oia.losses.meter_reason_losses import meter_reason_share_loss


def test_reason_share_loss_only_backpropagates_selected_factor() -> None:
    logits = torch.zeros(2, 21, requires_grad=True)
    output = {
        "reason_logits_candidate": logits,
        "reason_logits_final": logits,
        "reason_logits_global": logits,
        "reason_logits_local": logits,
        "reason_annotation_delta": logits,
    }
    target = torch.zeros_like(logits)
    target[:, 7] = 1.0
    confidence = torch.full_like(logits, 0.5)
    observability = torch.ones_like(logits)

    loss = meter_reason_share_loss(
        output,
        target,
        confidence,
        factor_id=7,
        observability=observability,
    )["total"]
    loss.backward()

    assert float(logits.grad[:, 7].abs().sum()) > 0.0
    assert float(logits.grad[:, :7].abs().sum()) == 0.0
    assert float(logits.grad[:, 8:].abs().sum()) == 0.0
