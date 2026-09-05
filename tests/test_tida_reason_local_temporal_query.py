import torch
import inspect

from fate_oia.models.tida_reason_local_temporal_query import (
    TIDAReasonLocalTemporalQuery,
    bounded_reason_delta,
)
from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader
from fate_oia.losses.tida_losses import reason_local_utility_calibration_loss
from fate_oia.engine.evaluate_tida_oia import branch_metrics, save_epoch_outputs
from fate_oia.engine.train_tida_oia import (
    calibrate_reason_local_deployment,
    train_locked_deployment_views,
)
from fate_oia.engine import train_tida_oia


def test_reason_local_bounded_delta_preserves_small_signal_ranking():
    raw = torch.tensor([-0.10, -0.01, 0.0, 0.01, 0.10], requires_grad=True)
    value = bounded_reason_delta(raw, cap=0.02)

    assert torch.all(value[1:] > value[:-1])
    assert float(value.abs().max()) < 0.02
    assert torch.allclose(value[3], torch.tensor(0.0002), atol=1e-6)
    value.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad > 0)


def test_reason_directional_summary_preserves_signed_temporal_order():
    module = TIDAReasonLocalTemporalQuery(dim=4, num_reasons=2, num_heads=2)
    states = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1).expand(1, 2, 5, 4)
    timestamps = torch.arange(5, dtype=torch.float32).view(1, 5)
    valid = torch.ones(1, 5, dtype=torch.bool)

    ordered = module._signed_directional_summary(states, timestamps, valid)
    reversed_summary = module._signed_directional_summary(
        states.flip(2), timestamps, valid
    )

    assert ordered.shape == (1, 2, 12)
    assert torch.allclose(ordered[..., :4], -reversed_summary[..., :4])
    assert torch.all(ordered[..., :4] > 0)


def test_reason_directional_readout_receives_gradient_on_first_update():
    torch.manual_seed(4)
    module = TIDAReasonLocalTemporalQuery(
        dim=8, num_reasons=3, num_heads=2, directional_enabled=True
    )
    output = module(
        torch.randn(2, 4, 3, 8),
        torch.randn(2, 3, 8),
        torch.arange(5, dtype=torch.float32).view(1, 5).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 3),
    )
    output["reason_local_candidate_logits"].sum().backward()
    grad = module.directional_readout_weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_reason_directional_readout_is_backward_compatible_by_default():
    module = TIDAReasonLocalTemporalQuery(dim=8, num_reasons=3, num_heads=2)

    assert module.directional_enabled is False
    assert module.directional_readout_weight.requires_grad is False


def test_reason_action_condition_is_zero_init_learnable_and_firewalled():
    torch.manual_seed(9)
    module = TIDAReasonLocalTemporalQuery(
        dim=8,
        num_reasons=3,
        num_heads=2,
        action_condition_enabled=True,
        num_actions=4,
    )
    action_delta = torch.randn(2, 4, requires_grad=True)
    output = module(
        torch.randn(2, 4, 3, 8),
        torch.randn(2, 3, 8),
        torch.arange(5, dtype=torch.float32).view(1, 5).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 3),
        action_condition_delta=action_delta,
        shuffled_action_condition_delta=action_delta.flip(1),
        selected_deleted_action_condition_delta=torch.zeros_like(action_delta),
        random_deleted_action_condition_delta=0.5 * action_delta,
    )

    assert torch.count_nonzero(output["reason_local_action_condition_delta"]) == 0
    output["reason_local_candidate_logits"].sum().backward()
    assert module.action_condition_weight.grad is not None
    assert module.action_condition_weight.grad.abs().sum() > 0
    assert action_delta.grad is None


def test_reason_action_condition_recomputes_counterfactual_branches():
    module = TIDAReasonLocalTemporalQuery(
        dim=8,
        num_reasons=3,
        num_heads=2,
        action_condition_enabled=True,
        num_actions=4,
    )
    with torch.no_grad():
        module.action_condition_weight.fill_(0.25)
    action_delta = torch.ones(1, 4)
    output = module(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 3, 8),
        torch.arange(5, dtype=torch.float32).view(1, 5),
        torch.ones(1, 5, dtype=torch.bool),
        image_logits=torch.randn(1, 3),
        action_condition_delta=action_delta,
        shuffled_action_condition_delta=-action_delta,
        selected_deleted_action_condition_delta=torch.zeros_like(action_delta),
        random_deleted_action_condition_delta=0.5 * action_delta,
    )

    assert not torch.allclose(
        output["reason_local_candidate_delta"],
        output["reason_local_shuffled_delta"],
    )
    assert not torch.allclose(
        output["reason_local_selected_deleted_delta"],
        output["reason_local_random_deleted_delta"],
    )


def test_reason_action_condition_preserves_small_signal_magnitude_while_bounded():
    module = TIDAReasonLocalTemporalQuery(
        dim=8,
        num_reasons=3,
        num_heads=2,
        action_condition_enabled=True,
        action_condition_cap=0.01,
        num_actions=4,
    )
    raw = torch.tensor([[0.005, -0.005, 1.0]])

    candidate = module._action_condition_candidate(
        raw,
        available=torch.ones_like(raw),
        temporal_scale=1.0,
    )

    assert candidate[0, 0] > 0.004
    assert candidate[0, 1] < -0.004
    assert candidate.abs().max() <= 0.01


def test_reason_action_token_condition_is_zero_init_learnable_and_firewalled():
    torch.manual_seed(17)
    module = TIDAReasonLocalTemporalQuery(
        dim=8,
        num_reasons=3,
        num_heads=2,
        action_condition_enabled=True,
        action_condition_mode="token",
        num_actions=4,
    )
    action_history = torch.randn(2, 4, 8, requires_grad=True)
    action_target = torch.randn(2, 4, 8, requires_grad=True)
    output = module(
        torch.randn(2, 4, 3, 8),
        torch.randn(2, 3, 8),
        torch.arange(5, dtype=torch.float32).view(1, 5).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 3),
        action_condition_history_summary=action_history,
        shuffled_action_condition_history_summary=action_history.flip(1),
        selected_deleted_action_condition_history_summary=torch.zeros_like(action_history),
        random_deleted_action_condition_history_summary=0.5 * action_history,
        action_condition_target_tokens=action_target,
    )

    assert torch.count_nonzero(output["reason_local_action_condition_delta"]) == 0
    assert output["reason_local_action_condition_attention"].shape == (2, 3, 4)
    assert torch.allclose(
        output["reason_local_action_condition_attention"].sum(-1),
        torch.ones(2, 3),
    )
    output["reason_local_candidate_logits"].sum().backward()
    grad = module.action_token_readout_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert action_history.grad is None
    assert action_target.grad is None


def test_reason_action_token_condition_recomputes_counterfactual_summaries():
    torch.manual_seed(23)
    module = TIDAReasonLocalTemporalQuery(
        dim=8,
        num_reasons=3,
        num_heads=2,
        action_condition_enabled=True,
        action_condition_mode="token",
        num_actions=4,
    )
    with torch.no_grad():
        module.action_token_readout_weight.normal_(std=0.1)
    action_history = torch.randn(1, 4, 8)
    output = module(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 3, 8),
        torch.arange(5, dtype=torch.float32).view(1, 5),
        torch.ones(1, 5, dtype=torch.bool),
        image_logits=torch.randn(1, 3),
        action_condition_history_summary=action_history,
        shuffled_action_condition_history_summary=-action_history,
        selected_deleted_action_condition_history_summary=torch.zeros_like(action_history),
        random_deleted_action_condition_history_summary=0.5 * action_history,
        action_condition_target_tokens=torch.randn(1, 4, 8),
    )

    assert not torch.allclose(
        output["reason_local_candidate_delta"],
        output["reason_local_shuffled_delta"],
    )
    assert not torch.allclose(
        output["reason_local_selected_deleted_delta"],
        output["reason_local_random_deleted_delta"],
    )


def test_query_reader_returns_target_private_reason_patch_reads():
    torch.manual_seed(3)
    reader = TIDATerminalQueryReader(
        dim=16, num_actions=4, num_predicates=5, layer_ids=(3, 7, 11)
    )
    field = torch.randn(2, 3, 12, 16)
    action = torch.randn(2, 4, 16)
    predicate = torch.randn(2, 5, 16)
    predicate_identity = torch.randn(5, 16)
    reason = torch.randn(2, 7, 16)

    output = reader(
        field, action, predicate, predicate_identity, grid_hw=(3, 4), reason_nodes=reason
    )

    assert output["reason_query_tokens"].shape == (2, 7, 16)
    assert output["reason_query_attention"].shape == (2, 7, 12)
    assert output["query_tokens"].shape == (2, 9, 16)


def test_reason_local_query_is_exact_image_fallback_but_candidate_gets_gradient():
    torch.manual_seed(5)
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=7, num_heads=4, cap=0.08, utility_open_prior=0.10
    )
    history = torch.randn(2, 4, 7, 16)
    target = torch.randn(2, 7, 16)
    timestamps = torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    image_logits = torch.randn(2, 7)

    output = module(history, target, timestamps, valid, image_logits=image_logits)

    assert torch.equal(output["reason_local_deploy_logits"], image_logits)
    assert torch.count_nonzero(output["reason_local_deploy_gate"]) == 0
    loss = (output["reason_local_candidate_logits"] - 1.0).square().mean()
    loss.backward()
    grad = module.reason_readout_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0


def test_reason_local_utility_receives_signed_and_magnitude_evidence():
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    assert module.utility[0].normalized_shape == (22,)


def test_reason_local_query_has_strict_zero_delta_without_history():
    torch.manual_seed(7)
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    history = torch.randn(2, 4, 7, 16)
    target = torch.randn(2, 7, 16)
    timestamps = torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    valid[:, -1] = True
    image_logits = torch.randn(2, 7)

    output = module(history, target, timestamps, valid, image_logits=image_logits)

    assert torch.count_nonzero(output["reason_local_candidate_delta"]) == 0
    assert torch.equal(output["reason_local_deploy_logits"], image_logits)


def test_reason_local_candidate_cannot_degenerate_to_constant_label_bias():
    torch.manual_seed(11)
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 7),
    )
    assert torch.count_nonzero(output["reason_local_candidate_delta"]) == 0


def test_reason_local_has_independent_per_label_readouts():
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 7),
    )
    assert module.reason_readout_weight.shape == (7, 16)
    loss = output["reason_local_candidate_logits"][:, 3].sum()
    loss.backward()
    grad = module.reason_readout_weight.grad
    assert float(grad[3].abs().sum()) > 0.0
    assert torch.count_nonzero(grad[:3]) == 0
    assert torch.count_nonzero(grad[4:]) == 0


def test_reason_local_reports_target_private_motion_energy():
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 7),
    )

    motion = output["reason_local_motion_energy"]
    assert motion.shape == (2, 1, 7)
    assert torch.isfinite(motion).all()
    assert torch.all((motion >= 0.0) & (motion <= 1.0))
    assert float(motion.std()) > 0.0


def test_reason_local_target_query_attends_only_valid_history_frames():
    torch.manual_seed(13)
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    valid = torch.tensor(
        [[True, True, False, False, True], [True, False, True, False, True]]
    )
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        valid,
        image_logits=torch.randn(2, 7),
    )

    attention = output["reason_local_temporal_attention"]
    assert attention.shape == (2, 7, 4)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 7), atol=1e-6)
    invalid = (~valid[:, :4])[:, None].expand_as(attention)
    assert torch.count_nonzero(attention.masked_select(invalid)) == 0


def test_reason_local_target_query_attention_is_label_specific_and_trainable():
    torch.manual_seed(17)
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    with torch.no_grad():
        module.reason_readout_weight.normal_(0.0, 0.1)
    shared_history = torch.randn(1, 4, 1, 16).expand(-1, -1, 7, -1).clone()
    output = module(
        shared_history,
        torch.randn(1, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]),
        torch.ones(1, 5, dtype=torch.bool),
        image_logits=torch.randn(1, 7),
    )

    attention = output["reason_local_temporal_attention"]
    assert float(attention.std(dim=1).sum()) > 0.0
    output["reason_local_candidate_logits"].sum().backward()
    assert module.temporal_query_proj.weight.grad is not None
    assert float(module.temporal_query_proj.weight.grad.abs().sum()) > 0.0


def test_reason_local_temporal_ontology_mask_keeps_static_labels_exactly_zero():
    torch.manual_seed(19)
    module = TIDAReasonLocalTemporalQuery(
        dim=16,
        num_reasons=7,
        num_heads=4,
        temporal_reason_indices=(1, 4),
    )
    with torch.no_grad():
        module.reason_readout_weight.normal_(0.0, 0.1)
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 7),
    )

    static = torch.tensor([0, 2, 3, 5, 6])
    assert torch.count_nonzero(
        output["reason_local_candidate_delta"][:, static]
    ) == 0
    assert torch.equal(
        output["reason_local_temporal_reason_mask"],
        torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_reason_local_train_calib_center_is_applied_only_with_history():
    torch.manual_seed(23)
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)
    center = torch.linspace(-0.01, 0.01, 7)
    module.set_deployment_policy(
        torch.ones(7), torch.ones(7), torch.zeros(7),
        center=center, source="train_calib_oof",
    )
    image_logits = torch.randn(2, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, :4] = False
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        valid,
        image_logits=image_logits,
    )

    assert torch.allclose(
        output["reason_local_centered_candidate_delta"][0],
        output["reason_local_candidate_delta"][0] - center,
    )
    assert torch.count_nonzero(
        output["reason_local_centered_candidate_delta"][1]
    ) == 0
    assert torch.equal(output["reason_local_deploy_logits"][1], image_logits[1])


def test_reason_local_deploy_policy_is_bounded_and_exactly_selective():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=7, num_heads=4, cap=0.02
    )
    with torch.no_grad():
        module.reason_readout_weight.normal_(0.0, 0.1)
    gate = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scale = torch.tensor([-0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    cutoff = torch.zeros(7)
    module.set_deployment_policy(gate, scale, cutoff, source="train_calib_oof")
    image_logits = torch.randn(2, 7)
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=image_logits,
    )

    assert torch.equal(output["reason_local_deploy_logits"][:, 1:], image_logits[:, 1:])
    assert torch.allclose(
        output["reason_local_deploy_delta"][:, 0],
        (-0.5 * output["reason_local_candidate_delta"][:, 0]).clamp(-0.02, 0.02),
    )
    assert output["reason_local_policy_source"] == "train_calib_oof"


def test_reason_local_negative_scale_inverts_raw_helpfulness_utility():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=7, num_heads=4, cap=0.02, utility_open_prior=0.10
    )
    with torch.no_grad():
        module.reason_readout_weight.normal_(0.0, 0.1)
    module.set_deployment_policy(
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.5] * 7),
        utility_inverted=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        source="train_calib_oof",
    )
    output = module(
        torch.randn(2, 4, 7, 16),
        torch.randn(2, 7, 16),
        torch.tensor([[-2.0, -1.0, -0.5, -0.2, 0.0]]).expand(2, -1),
        torch.ones(2, 5, dtype=torch.bool),
        image_logits=torch.randn(2, 7),
    )

    assert torch.all(output["reason_local_utility_probability"][:, 0] < 0.5)
    assert torch.count_nonzero(output["reason_local_deploy_gate"][:, 0]) == 2


def test_reason_local_policy_is_fit_from_train_calib_and_opens_only_supported_label():
    module = TIDAReasonLocalTemporalQuery(
        dim=16, num_reasons=7, num_heads=4, cap=0.02
    )

    class Model:
        reason_local_query_enabled = True
        reason_local_query = module

    count = 20
    reason_target = torch.zeros(count, 7)
    reason_target[:10, 0] = 1.0
    image_reason = torch.full((count, 7), -1.0)
    image_reason[:, 0] = -0.01
    candidate = torch.zeros_like(image_reason)
    candidate[:10, 0] = 0.02
    utility = torch.zeros_like(image_reason)
    utility[:10, 0] = 1.0
    rows = {
        "image_action": torch.zeros(count, 4),
        "video_action": torch.zeros(count, 4),
        "image_reason": image_reason,
        "video_reason": image_reason,
        "reason_local_candidate_delta": candidate,
        "reason_local_utility_probability": utility,
        "action_target": torch.zeros(count, 4),
        "reason_target": reason_target,
    }
    config = {
        "locked_image_thresholds": [0.5] * 11,
        "locked_image_threshold_source": "train_calib_fixture",
        "reason_local_policy_scales": [-1.0, 0.0, 1.0],
        "reason_local_policy_cutoffs": [0.5],
        "reason_local_policy_oof_folds": 5,
    }

    fit = calibrate_reason_local_deployment(Model(), rows, config)

    assert fit["test_labels_used"] is False
    assert fit["source"] == "train_calib_oof"
    assert module.deployment_label_gate[0] == 1
    assert module.deployment_scale[0] == 1
    assert torch.count_nonzero(module.deployment_label_gate[1:]) == 0
    assert torch.allclose(
        module.deployment_center,
        candidate.median(0).values,
    )
    assert fit["candidate_center_source"] == "train_calib_median"
    expected_centered = candidate - candidate.median(0).values[None]
    assert torch.allclose(rows["reason_local_centered_candidate_delta"], expected_centered)
    assert torch.allclose(
        rows["reason_local_centered_candidate"], image_reason + expected_centered
    )


def test_reason_local_policy_keeps_zero_fallback_for_proper_score_only_tie():
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)

    class Model:
        reason_local_query_enabled = True
        reason_local_query = module

    count = 20
    target = torch.zeros(count, 7)
    target[:10, 0] = 1.0
    logits = torch.full((count, 7), -1.0)
    logits[:10, 0] = 1.0
    candidate = torch.zeros_like(logits)
    candidate[:, 0] = 0.02
    utility = torch.zeros_like(logits)
    utility[:10, 0] = 1.0
    fit = calibrate_reason_local_deployment(
        Model(),
        {
            "image_action": torch.zeros(count, 4),
            "video_action": torch.zeros(count, 4),
            "image_reason": logits,
            "video_reason": logits,
            "reason_local_candidate_delta": candidate,
            "reason_local_utility_probability": utility,
            "action_target": torch.zeros(count, 4),
            "reason_target": target,
        },
        {
            "locked_image_thresholds": [0.5] * 11,
            "locked_image_threshold_source": "train_calib_fixture",
            "reason_local_policy_scales": [0.0, 1.0],
            "reason_local_policy_cutoffs": [0.5],
            "reason_local_policy_oof_folds": 5,
            "reason_local_policy_allow_proper_score_tie": False,
        },
    )

    assert fit["nll_improvement"][0] == 0.0
    assert torch.count_nonzero(module.deployment_label_gate) == 0


def test_reason_local_leave_source_out_records_single_source_fallback():
    module = TIDAReasonLocalTemporalQuery(dim=16, num_reasons=7, num_heads=4)

    class Model:
        reason_local_query_enabled = True
        reason_local_query = module

    count = 20
    rows = {
        "image_action": torch.zeros(count, 4),
        "video_action": torch.zeros(count, 4),
        "image_reason": torch.zeros(count, 7),
        "video_reason": torch.zeros(count, 7),
        "reason_local_candidate_delta": torch.zeros(count, 7),
        "reason_local_utility_probability": torch.zeros(count, 7),
        "action_target": torch.zeros(count, 4),
        "reason_target": torch.zeros(count, 7),
        "source_batches": ["one_source"] * count,
    }
    fit = calibrate_reason_local_deployment(
        Model(),
        rows,
        {
            "reason_local_policy_leave_source_out": True,
            "reason_local_policy_oof_folds": 5,
        },
    )
    assert fit["fold_strategy"] == "label_stratified_oof"
    assert fit["source_fallback_reason"] == "single_source_train_calib"


def test_training_saves_optimizer_boundary_checkpoint_before_evaluation():
    source = inspect.getsource(train_tida_oia.train)
    checkpoint = source.index('"checkpoint_pre_eval_latest.pth"')
    evaluation = source.index("model.eval()", checkpoint)
    assert checkpoint < evaluation


def test_reason_local_utility_ignores_unknown_negatives_but_learns_certified_signs():
    utility = torch.zeros(1, 3, requires_grad=True)
    candidate = torch.tensor([[0.10, -0.10, 0.10]])
    observed = torch.tensor([[1.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 1.0, 0.50]])

    loss = reason_local_utility_calibration_loss(
        utility, candidate, observed, contradiction
    )
    loss.backward()

    assert utility.grad[0, 0] < 0
    assert utility.grad[0, 1] < 0
    assert utility.grad[0, 2] == 0


def test_reason_local_utility_does_not_close_before_candidate_moves():
    utility = torch.zeros(1, 3, requires_grad=True)
    loss = reason_local_utility_calibration_loss(
        utility,
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.5]]),
    )
    loss.backward()
    assert torch.count_nonzero(utility.grad) == 0


def test_evaluation_exposes_candidate_legacy_and_safe_deploy_views():
    rows = {
        "image_action": torch.randn(4, 4),
        "video_action": torch.randn(4, 4),
        "image_reason": torch.randn(4, 21),
        "video_reason": torch.randn(4, 21),
        "legacy_video_reason": torch.randn(4, 21),
        "reason_local_candidate": torch.randn(4, 21),
        "reason_local_deploy": torch.randn(4, 21),
        "action_target": torch.randint(0, 2, (4, 4)).float(),
        "reason_target": torch.randint(0, 2, (4, 21)).float(),
    }
    metrics = branch_metrics(rows)
    assert set(metrics) == {
        "image", "video", "legacy_reason_route",
        "reason_local_candidate", "reason_local_deploy",
    }


def test_evaluation_exposes_independent_action_conditioned_reason_view():
    rows = {
        "image_action": torch.randn(8, 4),
        "video_action": torch.randn(8, 4),
        "image_reason": torch.randn(8, 21),
        "video_reason": torch.randn(8, 21),
        "reason_local_action_condition": torch.randn(8, 21),
        "action_target": torch.randint(0, 2, (8, 4)).float(),
        "reason_target": torch.randint(0, 2, (8, 21)).float(),
    }

    metrics = branch_metrics(rows)

    assert "reason_local_action_condition" in metrics
    assert "Exp_mAP" in metrics["reason_local_action_condition"]


def test_train_locked_views_expose_action_token_direct_candidates():
    rows = {
        "image_action": torch.randn(8, 4),
        "video_action": torch.randn(8, 4),
        "action_local_candidate": torch.randn(8, 4),
        "image_reason": torch.randn(8, 21),
        "video_reason": torch.randn(8, 21),
        "reason_local_action_condition": torch.randn(8, 21),
        "action_target": torch.randint(0, 2, (8, 4)).float(),
        "reason_target": torch.randint(0, 2, (8, 21)).float(),
    }
    thresholds = {
        "image": torch.full((25,), 0.5),
        "video": torch.full((25,), 0.5),
    }

    views = train_locked_deployment_views(rows, thresholds)

    assert "action_local_direct_candidate" in views
    assert "reason_local_action_condition_direct_candidate" in views
    assert "action_token_joint_direct_candidate" in views
    assert views["action_token_joint_direct_candidate"]["Act_mAP"] >= 0.0
    assert views["action_token_joint_direct_candidate"]["Exp_mAP"] >= 0.0


def test_epoch_artifacts_save_independent_action_conditioned_reason_tensors(tmp_path):
    rows = {
        "file_names": ["sample.jpg"],
        "reason_local_action_condition": torch.randn(1, 21),
        "reason_local_action_condition_delta": torch.randn(1, 21),
        "reason_local_action_condition_attention": torch.rand(1, 21, 4),
    }

    save_epoch_outputs(
        tmp_path,
        0,
        rows,
        metrics={},
        thresholds={},
        mechanism={},
    )

    epoch_dir = tmp_path / "epoch_000"
    assert (epoch_dir / "reason_local_action_condition_test.pt").is_file()
    assert (epoch_dir / "reason_local_action_condition_delta_test.pt").is_file()
    assert (epoch_dir / "reason_local_action_condition_attention_test.pt").is_file()
