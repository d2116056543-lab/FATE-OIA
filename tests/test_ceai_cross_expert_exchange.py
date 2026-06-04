import torch

from fate_oia.models.ceai_cross_expert_exchange import ControlledCrossExpertExchange


def test_cross_exchange_stopgrad_and_r2a_disable():
    exchange = ControlledCrossExpertExchange(dim=24, action_dim=4, reason_dim=21, heads=4)
    action_tokens = torch.randn(2, 4, 24, requires_grad=True)
    reason_tokens = torch.randn(2, 21, 24, requires_grad=True)
    q_ar = torch.ones(2, 4, 21) * 0.8
    out_off = exchange(action_tokens, reason_tokens, q_ar=q_ar, readiness={"r2a_active": False})
    assert out_off["action_tokens"].shape == action_tokens.shape
    assert out_off["reason_tokens"].shape == reason_tokens.shape
    assert out_off["stats"]["r2a_active_rate"] == 0.0
    loss = out_off["reason_tokens"].sum()
    loss.backward(retain_graph=True)
    assert action_tokens.grad is None or action_tokens.grad.abs().sum() == 0
    out_on = exchange(action_tokens.detach(), reason_tokens.detach(), q_ar=q_ar, readiness={"r2a_active": True})
    assert out_on["stats"]["r2a_gate_mean"] > 0
    low = exchange(action_tokens.detach(), reason_tokens.detach(), q_ar=q_ar * 0.05, readiness={"r2a_active": True})
    assert out_on["stats"]["action_token_delta_norm"] > low["stats"]["action_token_delta_norm"]
