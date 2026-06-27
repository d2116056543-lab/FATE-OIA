from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.decision_ledger import DecisionLedgerHead


def test_benefit_gate_has_detached_advantage_target_and_gradients() -> None:
    ledger = DecisionLedgerHead(dim=16, num_actions=3)
    visual = torch.randn(2, 16)
    motion = torch.randn(2, 16)
    predicate = torch.randn(2, 16)
    factors = torch.randn(2, 5, 16)
    flow = torch.randn(2, 5, 3)
    soft = torch.softmax(torch.randn(2, 3), dim=-1)

    out = ledger(visual, motion, predicate, factors, flow, action_soft_target=soft)
    assert out.benefit_target is not None
    assert out.benefit_target.requires_grad is False

    assert out.benefit_target.shape == (2, 5, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.benefit_gate_logits.mean(-1, keepdim=True), out.benefit_target)
    loss.backward()
    assert ledger.benefit_gate.weight.grad is not None
    assert torch.isfinite(ledger.benefit_gate.weight.grad).all()


def test_benefit_gate_advantage_loss_is_autocast_safe() -> None:
    from fate_oia.losses.acpr_interactflow_losses import compute_interactflow_losses

    class Batch:
        pass

    model = __import__("fate_oia.acpr_interactflow.model", fromlist=["ACPRInteractFlowPPModel"]).ACPRInteractFlowPPModel(
        use_mock_dino=True,
        action_dim=3,
        dino_input_height=64,
        dino_input_width=96,
        dino_chunk_size=2,
    )
    frames = torch.rand(1, 15, 3, 64, 96)
    batch = Batch()
    batch.action_soft = torch.softmax(torch.randn(1, 3), dim=-1)
    batch.action_majority = torch.tensor([1])
    batch.exp29 = torch.zeros(1, 29)
    batch.exp29[:, 0] = 1.0
    batch.exp29_mask = torch.ones(1, 29)
    batch.paper_effective_weight = torch.ones(1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        output = model(frames, action_soft_target=batch.action_soft)
        loss, terms = compute_interactflow_losses(output, batch, weights={"benefit_gate_advantage_bce": 0.04})
    assert torch.isfinite(loss)
    assert torch.isfinite(terms["benefit_gate_advantage_bce"])
