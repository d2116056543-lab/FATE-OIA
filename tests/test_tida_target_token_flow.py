import torch

from fate_oia.models.tida_target_token_flow import TIDATargetTokenFlow


def _inputs(labels: int = 4, dim: int = 16):
    torch.manual_seed(7)
    history = torch.randn(2, 4, labels, dim)
    terminal = history[:, -1] + 0.2 * torch.randn(2, labels, dim)
    logits = torch.randn(2, labels)
    timestamps = torch.tensor([[-4.0, -3.0, -2.0, -1.0, 0.0]]).expand(2, -1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    return history, terminal, logits, timestamps, valid


def test_target_token_flow_is_exact_zero_effect_at_initialization_and_bounded():
    reader = TIDATargetTokenFlow(num_labels=4, dim=16, hidden_dim=16, cap=0.05)
    history, terminal, logits, timestamps, valid = _inputs()

    initial = reader(history, terminal, timestamps, valid, base_logits=logits)

    assert initial["predicted_terminal_token"].shape == terminal.shape
    assert initial["ordered_prediction_error"].shape == logits.shape
    assert initial["reversed_prediction_error"].shape == logits.shape
    assert initial["repeated_prediction_error"].shape == logits.shape
    assert initial["shuffled_prediction_error"].shape == logits.shape
    assert torch.count_nonzero(initial["candidate_delta"]) == 0
    assert torch.equal(initial["deploy_logits"], logits)
    assert all(
        torch.isfinite(value).all()
        for value in initial.values()
        if isinstance(value, torch.Tensor)
    )

    with torch.no_grad():
        reader.candidate_output_weight.fill_(2.0)
    learned = reader(history, terminal, timestamps, valid, base_logits=logits)
    assert torch.all(learned["candidate_delta"].abs() <= 0.05 + 1e-7)


def test_target_token_flow_uses_time_order_and_returns_control_predictions():
    reader = TIDATargetTokenFlow(num_labels=4, dim=16, hidden_dim=16, cap=0.05)
    history, terminal, logits, timestamps, valid = _inputs()

    ordered = reader(history, terminal, timestamps, valid, base_logits=logits)
    reversed_history = reader(
        history.flip(1), terminal, timestamps, valid, base_logits=logits
    )

    assert not torch.allclose(
        ordered["predicted_terminal_token"],
        reversed_history["predicted_terminal_token"],
    )
    assert not torch.allclose(
        ordered["predicted_terminal_token"], ordered["reversed_predicted_terminal_token"]
    )
    assert not torch.allclose(
        ordered["predicted_terminal_token"], ordered["repeated_predicted_terminal_token"]
    )
    assert not torch.allclose(
        ordered["predicted_terminal_token"], ordered["shuffled_predicted_terminal_token"]
    )


def test_target_token_flow_ignores_missing_history_and_keeps_owner_isolation():
    action = TIDATargetTokenFlow(num_labels=4, dim=16, hidden_dim=16, cap=0.05)
    reason = TIDATargetTokenFlow(num_labels=21, dim=16, hidden_dim=16, cap=0.04)
    history, terminal, logits, timestamps, valid = _inputs()
    reason_history, reason_terminal, reason_logits, _, _ = _inputs(labels=21)

    no_history = valid.clone()
    no_history[:, :-1] = False
    empty = action(history, terminal, timestamps, no_history, base_logits=logits)
    assert torch.count_nonzero(empty["candidate_delta"]) == 0
    assert not empty["history_available"].any()

    with torch.no_grad():
        action.candidate_output_weight.fill_(0.25)
        reason.candidate_output_weight.fill_(0.25)
    action(
        history, terminal, timestamps, valid, base_logits=logits
    )["candidate_delta"].sum().backward()

    assert any(parameter.grad is not None for parameter in action.parameters())
    assert all(parameter.grad is None for parameter in reason.parameters())
    assert reason(
        reason_history,
        reason_terminal,
        timestamps,
        valid,
        base_logits=reason_logits,
    )["candidate_delta"].shape == (2, 21)


def test_target_token_flow_does_not_backpropagate_into_frozen_token_sources():
    reader = TIDATargetTokenFlow(num_labels=4, dim=16, hidden_dim=16, cap=0.05)
    history, terminal, logits, timestamps, valid = _inputs()
    history.requires_grad_(True)
    terminal.requires_grad_(True)
    logits.requires_grad_(True)

    output = reader(history, terminal, timestamps, valid, base_logits=logits)
    output["predicted_terminal_token"].sum().backward()

    assert history.grad is None
    assert terminal.grad is None
    assert logits.grad is None
