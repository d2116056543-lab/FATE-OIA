import torch

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_terminal_predictor_uses_query_identity_not_target_static_context():
    module = TIDATerminalInnovation(dim=8).eval()
    query_identity = torch.randn(2, 4, 8)
    history = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    first = module(query_identity, history, target, torch.ones(2, dtype=torch.bool))
    second = module(query_identity, history, target * 100.0, torch.ones(2, dtype=torch.bool))

    assert torch.equal(first["terminal_prediction_history"], second["terminal_prediction_history"])
    assert torch.equal(first["terminal_prediction_no_history"], second["terminal_prediction_no_history"])
    assert first["terminal_prediction_history"].shape == target.shape


def test_no_history_reconstruction_is_marked_diagnostic_only():
    module = TIDATerminalInnovation(dim=8)
    result = module(
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
        torch.ones(1, dtype=torch.bool),
    )
    assert result["terminal_no_history_optimized"].item() is False

