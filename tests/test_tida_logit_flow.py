import torch

from fate_oia.models.tida_logit_flow import TIDALogitFlowReader


def _inputs(labels=4):
    history = torch.tensor(
        [
            [[-0.8] * labels, [-0.4] * labels, [0.1] * labels, [0.5] * labels],
            [[0.7] * labels, [0.4] * labels, [0.2] * labels, [-0.1] * labels],
        ],
        dtype=torch.float32,
    )
    terminal = torch.tensor([[0.9] * labels, [-0.3] * labels])
    timestamps = torch.tensor([[-4.0, -3.0, -2.0, -1.0, 0.0]]).expand(2, -1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    return history, terminal, timestamps, valid


def test_logit_flow_is_exact_zero_effect_at_initialization_and_bounded_after_learning():
    reader = TIDALogitFlowReader(num_labels=4, hidden_dim=16, cap=0.05)
    history, terminal, timestamps, valid = _inputs()

    initial = reader(history, terminal, timestamps, valid)

    assert torch.count_nonzero(initial["candidate_delta"]) == 0
    assert torch.equal(initial["deploy_logits"], terminal)
    with torch.no_grad():
        reader.candidate_output.weight.fill_(2.0)
    learned = reader(history, terminal, timestamps, valid)
    assert torch.all(learned["candidate_delta"].abs() <= 0.05 + 1e-7)


def test_logit_flow_is_time_order_sensitive_and_ignores_invalid_history():
    reader = TIDALogitFlowReader(num_labels=4, hidden_dim=16, cap=0.05)
    history, terminal, timestamps, valid = _inputs()
    with torch.no_grad():
        reader.candidate_output.weight.fill_(0.25)

    ordered = reader(history, terminal, timestamps, valid)
    reversed_flow = reader(history.flip(1), terminal, timestamps, valid)
    no_history = reader(
        history, terminal, timestamps,
        valid & torch.tensor([False, False, False, False, True]),
    )

    assert not torch.equal(ordered["candidate_delta"], reversed_flow["candidate_delta"])
    assert torch.count_nonzero(no_history["candidate_delta"]) == 0
    assert not no_history["history_available"].any()


def test_independent_action_and_reason_logit_flow_readers_do_not_share_gradients():
    action = TIDALogitFlowReader(num_labels=4, hidden_dim=16, cap=0.05)
    reason = TIDALogitFlowReader(num_labels=21, hidden_dim=16, cap=0.04)
    action_history, action_terminal, timestamps, valid = _inputs(4)
    reason_history, reason_terminal, _, _ = _inputs(21)
    with torch.no_grad():
        action.candidate_output.weight.fill_(0.1)
        reason.candidate_output.weight.fill_(0.1)

    action(action_history, action_terminal, timestamps, valid)["candidate_delta"].sum().backward()

    assert any(parameter.grad is not None for parameter in action.parameters())
    assert all(parameter.grad is None for parameter in reason.parameters())
