import torch

from fate_oia.engine.tesa_diagnostics import schema_token_mismatch


def test_schema_corruption_changes_association_not_batch_identity() -> None:
    token = torch.arange(2 * 21 * 3).view(2, 21, 3)
    corrupt = schema_token_mismatch(token)
    assert torch.equal(corrupt[:, 0], token[:, 1])
    assert torch.equal(corrupt[0], torch.roll(token[0], -1, 0))
