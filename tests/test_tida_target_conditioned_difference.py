import torch
from torch.nn import functional as F
import inspect

from fate_oia.models.tida_oia_model import TIDAOIAModel
from fate_oia.models.tida_target_token_flow import TIDATargetTokenFlow


def _inputs(labels: int = 4, dim: int = 16):
    torch.manual_seed(29)
    history = torch.randn(3, 5, labels, dim)
    terminal = history[:, -1] + 0.15 * torch.randn(3, labels, dim)
    logits = torch.randn(3, labels)
    timestamps = torch.linspace(-5.0, 0.0, 6).expand(3, -1)
    valid = torch.ones(3, 6, dtype=torch.bool)
    target = torch.randint(0, 2, logits.shape).float()
    return history, terminal, logits, timestamps, valid, target


def _reader() -> TIDATargetTokenFlow:
    return TIDATargetTokenFlow(
        num_labels=4,
        dim=16,
        hidden_dim=16,
        cap=0.05,
        direct_difference_enabled=True,
    )


def test_direct_difference_is_zero_initialized_and_exposes_real_controls():
    reader = _reader()
    history, terminal, logits, timestamps, valid, _ = _inputs()

    output = reader(history, terminal, timestamps, valid, base_logits=logits)

    assert torch.equal(output["candidate_logits"], logits)
    assert torch.count_nonzero(output["candidate_delta"]) == 0
    assert output["repeated_candidate_delta"].shape == logits.shape
    assert output["shuffled_candidate_delta"].shape == logits.shape
    assert output["direct_temporal_summary"].shape == (3, 4, 16)

    with torch.no_grad():
        reader.direct_output_weight.normal_(std=0.2)
    learned = reader(history, terminal, timestamps, valid, base_logits=logits)
    assert torch.count_nonzero(learned["candidate_delta"]) > 0
    assert not torch.allclose(
        learned["candidate_delta"], learned["repeated_candidate_delta"]
    )
    assert not torch.allclose(
        learned["candidate_delta"], learned["shuffled_candidate_delta"]
    )


def test_direct_difference_zeroes_static_clips_and_has_two_step_gradients():
    reader = _reader()
    history, terminal, logits, timestamps, valid, target = _inputs()
    static_token = terminal[:, None].expand_as(history)
    static = reader(static_token, terminal, timestamps, valid, base_logits=logits)
    assert torch.count_nonzero(static["candidate_delta"]) == 0

    optimizer = torch.optim.SGD(reader.parameters(), lr=0.1)
    first = reader(history, terminal, timestamps, valid, base_logits=logits)
    F.binary_cross_entropy_with_logits(first["candidate_logits"], target).backward()
    assert reader.direct_output_weight.grad is not None
    assert torch.count_nonzero(reader.direct_output_weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = reader(history, terminal, timestamps, valid, base_logits=logits)
    F.binary_cross_entropy_with_logits(second["candidate_logits"], target).backward()
    assert reader.direct_input_projection.weight.grad is not None
    assert torch.count_nonzero(reader.direct_input_projection.weight.grad) > 0


def test_direct_difference_ignores_unavailable_history():
    reader = _reader()
    history, terminal, logits, timestamps, valid, _ = _inputs()
    valid[:, :-1] = False

    output = reader(history, terminal, timestamps, valid, base_logits=logits)

    assert torch.equal(output["candidate_logits"], logits)
    assert torch.count_nonzero(output["candidate_delta"]) == 0


def test_direct_difference_control_centering_removes_shared_video_bias():
    common = torch.randn(3, 4)
    ordered, repeated, shuffled = TIDATargetTokenFlow._control_center_scores(
        common, common, common
    )

    assert torch.count_nonzero(ordered) == 0
    assert torch.count_nonzero(repeated) == 0
    assert torch.count_nonzero(shuffled) == 0

    ordered_raw = torch.randn(3, 4)
    repeated_raw = torch.randn(3, 4)
    shuffled_raw = torch.randn(3, 4)
    ordered, repeated, shuffled = TIDATargetTokenFlow._control_center_scores(
        ordered_raw, repeated_raw, shuffled_raw
    )
    assert torch.allclose(
        ordered + repeated + shuffled, torch.zeros_like(ordered), atol=1e-6
    )


def test_direct_difference_shuffle_destroys_local_forward_order():
    index = TIDATargetTokenFlow._destructive_shuffle_index(14, torch.device("cpu"))

    assert sorted(index.tolist()) == list(range(14))
    forward_neighbors = (index[1:] - index[:-1]).eq(1).sum()
    assert int(forward_neighbors) == 0


def test_model_constructor_surfaces_direct_difference_mode():
    parameters = inspect.signature(TIDAOIAModel.__init__).parameters
    assert "target_token_direct_difference_enabled" in parameters
