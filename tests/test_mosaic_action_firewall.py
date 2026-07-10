from __future__ import annotations

import inspect

import torch

from fate_oia.models.mosaic_action_decoder import MOSAICActionDecoder


def _pyramid(batch: int = 2, dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch, dim, 45, 80),
        "F_mid": torch.randn(batch, dim, 23, 40),
        "F_ctx": torch.randn(batch, dim, 12, 20),
    }


def test_action_decoder_is_action_owned_and_fusion_equation_is_exact() -> None:
    decoder = MOSAICActionDecoder(num_states=8, dim=8, highres_topk=16, midres_topk=8, self_attention_heads=2)
    state_prob = torch.rand(2, 8)
    state_uncertainty = torch.rand(2, 8)
    output = decoder(_pyramid(), state_prob, state_uncertainty, state_gate_cap=0.25)

    assert set(output) == {
        "action_visual_logits",
        "action_state_logits",
        "action_state_gate",
        "action_logits_raw",
        "action_nodes",
        "action_state_attention",
    }
    assert output["action_visual_logits"].shape == (2, 4)
    assert output["action_nodes"].shape == (2, 4, 8)
    assert torch.allclose(
        output["action_logits_raw"],
        output["action_visual_logits"] + output["action_state_gate"] * output["action_state_logits"],
    )
    assert output["action_state_gate"].min() >= 0
    assert output["action_state_gate"].max() <= 0.25


def test_action_state_gate_strictly_falls_with_higher_state_uncertainty() -> None:
    torch.manual_seed(41)
    decoder = MOSAICActionDecoder(num_states=8, dim=8, highres_topk=16, midres_topk=8, self_attention_heads=2).eval()
    pyramid = _pyramid()
    state_prob = torch.full((2, 8), 0.7)
    low = decoder(pyramid, state_prob, torch.zeros(2, 8), state_gate_cap=0.25)
    high = decoder(pyramid, state_prob, torch.ones(2, 8), state_gate_cap=0.25)
    assert torch.all(high["action_state_gate"] <= low["action_state_gate"])
    assert torch.any(high["action_state_gate"] < low["action_state_gate"])


def test_action_gate_zero_recovers_visual_branch_exactly() -> None:
    decoder = MOSAICActionDecoder(num_states=8, dim=8, highres_topk=16, midres_topk=8, self_attention_heads=2)
    output = decoder(_pyramid(), torch.rand(2, 8), torch.rand(2, 8), state_gate_cap=0.0)
    assert torch.count_nonzero(output["action_state_gate"]) == 0
    assert torch.equal(output["action_logits_raw"], output["action_visual_logits"])


def test_action_forward_signature_and_source_have_a_hard_reason_firewall() -> None:
    signature = inspect.signature(MOSAICActionDecoder.forward)
    forbidden = {"reason", "reason_logits", "reason_nodes", "posterior", "propensity", "action_set"}
    assert forbidden.isdisjoint(signature.parameters)
    source = inspect.getsource(MOSAICActionDecoder)
    assert "reason_logits" not in source
    assert "reason_nodes" not in source
    assert "action_set" not in source

