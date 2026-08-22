import torch

from fate_oia.models.tida_oia_model import confidence_aware_reason_delta


def test_confidence_gate_keeps_positive_delta_and_selectively_suppresses_negative_delta():
    image = torch.tensor([[4.0, -4.0, 0.0]])
    delta = torch.tensor([[-1.0, -1.0, 0.5]], requires_grad=True)
    safe = confidence_aware_reason_delta(image, delta, temperature=0.5)
    torch.testing.assert_close(safe[:, 2], delta[:, 2])
    assert safe[0, 1] < safe[0, 0] < safe[0, 2]
    assert abs(float(safe[0, 1])) > 0.99
    assert abs(float(safe[0, 0])) < 0.001
    safe.sum().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()


def test_confidence_gate_rejects_invalid_inputs():
    try:
        confidence_aware_reason_delta(torch.zeros(1, 2), torch.zeros(1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was accepted")
