import torch
import inspect

from fate_oia.models.tida_reason_local_temporal_query import TIDAReasonLocalTemporalQuery
from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader
from fate_oia.losses.tida_losses import reason_local_utility_calibration_loss
from fate_oia.engine.evaluate_tida_oia import branch_metrics
from fate_oia.engine.train_tida_oia import calibrate_reason_local_deployment
from fate_oia.engine import train_tida_oia


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
    image_reason[:10, 0] = 0.01
    image_reason[10:, 0] = 0.01
    candidate = torch.zeros_like(image_reason)
    candidate[:, 0] = 0.02
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
    assert module.deployment_scale[0] == -1
    assert torch.count_nonzero(module.deployment_label_gate[1:]) == 0


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
    assert fit["fold_strategy"] == "random_oof"
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
