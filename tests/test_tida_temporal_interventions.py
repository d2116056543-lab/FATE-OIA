import torch

from fate_oia.utils.tida_temporal_interventions import apply_query_intervention


def test_repeated_last_uses_last_valid_history_token():
    tokens = torch.arange(1 * 4 * 2 * 1, dtype=torch.float32).reshape(1, 4, 2, 1)
    valid = torch.tensor([[True, True, False, False]])
    out = apply_query_intervention(tokens, "repeated_last", history_valid=valid)
    assert torch.equal(out, tokens[:, 1:2].expand_as(tokens))


def test_time_reverse_only_reorders_history():
    tokens = torch.randn(2, 5, 3, 4)
    out = apply_query_intervention(tokens, "time_reverse")
    assert torch.equal(out, tokens.flip(1))


def test_wrong_action_route_rotates_only_action_queries():
    tokens = torch.arange(1 * 3 * 7 * 2, dtype=torch.float32).reshape(1, 3, 7, 2)
    out = apply_query_intervention(tokens, "wrong_action_route", action_count=4)
    assert torch.equal(out[:, :, :4], tokens[:, :, :4].roll(1, dims=2))
    assert torch.equal(out[:, :, 4:], tokens[:, :, 4:])
