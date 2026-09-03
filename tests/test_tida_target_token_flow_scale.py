from types import SimpleNamespace

import torch

from fate_oia.models.tida_oia_model import TIDAOIAModel


class _FlowProbe:
    def __init__(self):
        self.scales = []

    def __call__(self, *args, base_logits, temporal_scale):
        self.scales.append(float(temporal_scale))
        delta = torch.ones_like(base_logits) * float(temporal_scale)
        return {"candidate_delta": delta, "deploy_delta": delta}


def test_independent_target_token_scale_is_not_suppressed_by_legacy_foundation():
    action = _FlowProbe()
    reason = _FlowProbe()
    model = SimpleNamespace(
        target_token_flow_enabled=True,
        target_token_flow_independent_scale=True,
        target_token_action=action,
        target_token_reason=reason,
    )
    output = {
        "video_action_logits": torch.zeros(2, 4),
        "video_action_logits_base": torch.zeros(2, 4),
        "video_reason_logits": torch.zeros(2, 21),
        "action_temporal_delta": torch.zeros(2, 4),
        "reason_temporal_delta": torch.zeros(2, 21),
    }
    tokens = torch.zeros(2, 3, 4, 8)
    terminal = torch.zeros(2, 4, 8)

    result = TIDAOIAModel._apply_target_token_flow(
        model,
        output,
        tokens,
        terminal,
        torch.zeros(2, 3, 21, 8),
        torch.zeros(2, 21, 8),
        torch.zeros(2, 4),
        torch.ones(2, 4, dtype=torch.bool),
        temporal_action_scale=0.0,
        temporal_reason_scale=0.0,
    )

    assert action.scales == [1.0]
    assert reason.scales == [1.0]
    assert torch.count_nonzero(result["target_token_action_candidate_delta"]) > 0
    assert torch.count_nonzero(result["target_token_reason_candidate_delta"]) > 0
