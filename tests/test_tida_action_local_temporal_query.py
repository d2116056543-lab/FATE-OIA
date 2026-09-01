import torch

from fate_oia.models.tida_action_local_temporal_query import (
    TIDAActionLocalTemporalQuery,
)


def _inputs(batch: int = 2):
    torch.manual_seed(17)
    history = torch.randn(batch, 3, 4, 32)
    target = torch.randn(batch, 4, 32)
    timestamps = torch.tensor([[-3.0, -1.5, -0.5, 0.0]]).expand(batch, -1)
    valid = torch.ones(batch, 4, dtype=torch.bool)
    logits = torch.randn(batch, 4)
    return history, target, timestamps, valid, logits


def test_action_local_query_is_exact_fallback_but_candidate_is_trainable():
    module = TIDAActionLocalTemporalQuery(dim=32, num_heads=4, cap=0.08)
    history, target, timestamps, valid, logits = _inputs()
    output = module(
        history, target, timestamps, valid, image_logits=logits
    )

    assert torch.equal(output["action_local_deploy_logits"], logits)
    assert output["action_local_candidate_delta"].shape == (2, 4)
    assert output["action_local_temporal_attention"].shape == (2, 4, 3)
    output["action_local_candidate_logits"].sum().backward()
    assert module.action_readout_weight.grad.abs().sum() > 0


def test_action_local_query_has_order_and_deletion_counterfactuals():
    module = TIDAActionLocalTemporalQuery(dim=32, num_heads=4, cap=0.08)
    history, target, timestamps, valid, logits = _inputs()
    with torch.no_grad():
        module.action_readout_weight.normal_(std=0.1)
    output = module(
        history, target, timestamps, valid, image_logits=logits
    )

    for key in (
        "action_local_shuffled_delta",
        "action_local_selected_deleted_delta",
        "action_local_random_deleted_delta",
        "action_local_selected_minus_random_gap",
        "action_local_motion_energy",
    ):
        assert key in output
        assert torch.isfinite(output[key]).all()
    assert not torch.equal(
        output["action_local_candidate_delta"],
        output["action_local_shuffled_delta"],
    )


def test_action_local_deployment_policy_is_bounded_and_selective():
    module = TIDAActionLocalTemporalQuery(dim=32, num_heads=4, cap=0.04)
    history, target, timestamps, valid, logits = _inputs()
    with torch.no_grad():
        module.action_readout_weight.normal_(std=0.2)
    module.set_deployment_policy(
        gate=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        scale=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        cutoff=torch.zeros(4),
        source="train_calib_oof",
    )
    output = module(
        history, target, timestamps, valid, image_logits=logits
    )

    assert torch.equal(output["action_local_deploy_logits"][:, 1:], logits[:, 1:])
    assert output["action_local_deploy_delta"].abs().max() <= 0.04 + 1e-7
    assert output["action_local_policy_source"] == "train_calib_oof"
