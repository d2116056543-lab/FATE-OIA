import torch

from fate_oia.models.tida_reason_local_temporal_query import (
    TIDAReasonLocalTemporalQuery,
)


def _inputs():
    torch.manual_seed(7)
    batch, frames, reasons, dim = 2, 5, 3, 16
    history = torch.randn(batch, frames, reasons, dim)
    target = torch.randn(batch, reasons, dim)
    timestamps = torch.tensor(
        [[-5.0, -2.8, -1.25, -0.35, -0.08, 0.0]] * batch
    )
    valid = torch.ones(batch, frames + 1, dtype=torch.bool)
    logits = torch.randn(batch, reasons)
    return history, target, timestamps, valid, logits


def test_site_reason_event_reader_is_exact_zero_at_initialization_but_trainable():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=3, num_heads=4, temporal_reason_indices=(0, 1, 2)
    )
    history, target, timestamps, valid, logits = _inputs()

    output = module(
        history, target, timestamps, valid, image_logits=logits,
        temporal_scale=1.0,
    )

    assert torch.equal(output["reason_local_candidate_delta"], torch.zeros_like(logits))
    output["reason_local_candidate_delta"].sum().backward()
    assert module.reason_readout_weight.grad is not None
    assert torch.isfinite(module.reason_readout_weight.grad).all()
    assert module.reason_readout_weight.grad.abs().sum() > 0


def test_site_reason_events_expose_real_order_and_deletion_counterfactuals():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=3, num_heads=4, temporal_reason_indices=(0, 1, 2)
    )
    history, target, timestamps, valid, logits = _inputs()
    with torch.no_grad():
        module.reason_readout_weight.normal_(std=0.2)

    output = module(
        history, target, timestamps, valid, image_logits=logits,
        temporal_scale=1.0,
    )

    for key in (
        "reason_local_velocity_rms",
        "reason_local_acceleration_rms",
        "reason_local_shuffled_delta",
        "reason_local_selected_deleted_delta",
        "reason_local_random_deleted_delta",
        "reason_local_selected_minus_random_gap",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()
    assert output["reason_local_velocity_rms"].gt(0).all()
    assert output["reason_local_acceleration_rms"].gt(0).all()
    assert not torch.allclose(
        output["reason_local_candidate_delta"],
        output["reason_local_shuffled_delta"],
    )
    assert not torch.allclose(
        output["reason_local_selected_deleted_delta"],
        output["reason_local_random_deleted_delta"],
    )


def test_site_reason_events_fall_back_exactly_when_history_is_unavailable():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=3, num_heads=4, temporal_reason_indices=(0, 1, 2)
    )
    history, target, timestamps, valid, logits = _inputs()
    valid[:, :-1] = False
    with torch.no_grad():
        module.reason_readout_weight.normal_(std=0.2)

    output = module(
        history, target, timestamps, valid, image_logits=logits,
        temporal_scale=1.0,
    )

    assert torch.equal(output["reason_local_candidate_delta"], torch.zeros_like(logits))
    assert torch.equal(output["reason_local_candidate_logits"], logits)
