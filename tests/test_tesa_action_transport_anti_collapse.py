import torch

from fate_oia.losses import meter_action_losses, meter_counterfactual_losses
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_counterfactual_losses import identity_corruption_loss
from fate_oia.models.meter_schema import default_meter_factor_schema
from fate_oia.models.meter_semantic_action import FactorSpecificActionTransport
from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead


def _transport() -> FactorSpecificActionTransport:
    torch.manual_seed(7)
    module = FactorSpecificActionTransport(dim=8, factor_dim=4, rank=3)
    with torch.no_grad():
        module.action_factor_compatibility.copy_(
            torch.tensor([[20.0, -20.0, 20.0, -20.0]] * 4)
        )
    return module


def test_action_loss_reaches_learnable_action_factor_compatibility() -> None:
    module = _transport()
    assert hasattr(module, "action_factor_compatibility")
    output = module(
        torch.zeros(2, 4),
        torch.randn(2, 4, 8),
        torch.randn(2, 4, 8),
        torch.ones(2, 4),
        torch.ones(4),
        factor_source=torch.ones(2, 4),
        progress=1.0,
        update_running_stats=True,
    )
    meter_action_loss(output, torch.tensor([[1.0, 0.0, 0.0, 1.0]] * 2))["total"].backward()
    assert module.action_factor_compatibility.grad is not None
    assert module.action_factor_compatibility.grad.abs().sum() > 0


def test_cap_uses_live_sparse_support_and_excludes_zero_source_factors() -> None:
    module = _transport()
    visual = torch.full((1, 4), 2.0)
    common = dict(
        action_logits_visual=visual,
        action_nodes=torch.randn(1, 4, 8),
        factor_typed_token=torch.randn(1, 4, 8),
        factor_reliability=torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        factor_action_ownership=torch.ones(4),
        progress=1.0,
        update_running_stats=True,
    )
    one = module(**common, factor_source=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    two = module(**common, factor_source=torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert one["action_effective_support_count"].eq(1).all()
    assert two["action_effective_support_count"].eq(2).all()
    assert torch.all(two["action_per_factor_kappa"] < one["action_per_factor_kappa"])
    assert one["action_factor_contributions"][..., 1:].eq(0).all()
    assert two["action_factor_contributions"][..., [1, 3]].eq(0).all()
    assert torch.all(one["action_evidence_delta"].abs() <= one["action_correction_kappa"] + 1e-6)
    assert torch.all(two["action_evidence_delta"].abs() <= two["action_correction_kappa"] + 1e-6)


def test_formal_transport_uses_predicted_observability_not_ground_truth() -> None:
    """The formal caller must pass factor_observability predicted by the model.

    Ground-truth observability belongs only to weak supervision/audit.  This
    test intentionally supplies a model-like probability tensor so a zero
    predicted observation blocks the factor even when its reliability is high.
    """
    module = _transport()
    predicted_factor_observability = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    output = module(
        torch.ones(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.ones(1, 4),
        torch.ones(4),
        factor_source=predicted_factor_observability,
        progress=1.0,
        update_running_stats=True,
    )
    assert output["action_factor_source_mask"][..., [1, 3]].eq(0).all()
    assert output["action_factor_contributions"][..., [1, 3]].eq(0).all()


def test_tiny_nonzero_observability_is_not_an_available_source() -> None:
    module = _transport()
    output = module(
        torch.zeros(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.ones(1, 4),
        torch.ones(4),
        factor_source=torch.full((1, 4), 1e-4),
        progress=1.0,
    )
    assert output["action_factor_source_mask"].eq(0).all()
    assert output["action_factor_contributions"].eq(0).all()


def test_high_entropy_reliability_does_not_make_observed_route_unreachable() -> None:
    module = _transport()
    output = module(
        torch.zeros(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.full((1, 4), 1e-6),
        torch.ones(4),
        factor_source=torch.ones(1, 4),
        progress=1.0,
    )
    assert output["action_factor_source_mask"].sum() > 0
    assert output["action_factor_effective_reliability"].min() >= 0.10
    assert output["action_factor_raw_contributions"].abs().sum() > 0


def test_source_aware_anti_monopoly_only_penalizes_multi_source_dominance() -> None:
    assert hasattr(meter_action_losses, "action_transport_anti_monopoly_loss")
    loss_fn = meter_action_losses.action_transport_anti_monopoly_loss
    single = loss_fn(
        torch.tensor([[[1.0, 0.0, 0.0]]]), torch.tensor([[[1.0, 0.0, 0.0]]])
    )
    no_source = loss_fn(
        torch.tensor([[[1.0, 0.0, 0.0]]]), torch.zeros(1, 1, 3)
    )
    dominant = loss_fn(
        torch.tensor([[[0.99, 0.01, 0.0]]]), torch.tensor([[[1.0, 1.0, 0.0]]])
    )
    balanced = loss_fn(
        torch.tensor([[[0.60, 0.40, 0.0]]]), torch.tensor([[[1.0, 1.0, 0.0]]])
    )
    assert single == 0
    assert no_source == 0
    assert dominant > balanced
    assert balanced == 0


def test_identity_loss_keeps_per_action_failures_from_cancelling() -> None:
    clean = torch.tensor([[[0.30], [-0.10]]])
    corrupt = torch.tensor([[[0.10], [-0.30]]])
    target = torch.tensor([[1.0, 0.0]])
    loss = identity_corruption_loss(clean, corrupt, target, margin=0.02)
    assert loss > 0.05


def test_near_boundary_ranking_has_sample_specific_nontrivial_delta_gradient() -> None:
    assert hasattr(meter_counterfactual_losses, "near_boundary_delta_ranking_loss")
    visual = torch.tensor([[0.02, 2.0], [-0.03, -2.0]])
    delta = torch.zeros_like(visual, requires_grad=True)
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss = meter_counterfactual_losses.near_boundary_delta_ranking_loss(
        visual, delta, target, margin=0.04
    )
    loss.backward()
    assert loss > 0
    assert delta.grad is not None
    assert delta.grad[:, 0].abs().mean() > delta.grad[:, 1].abs().mean()


def test_action_specificity_resolved_once_and_legacy_default_is_compatible() -> None:
    output = {
        "action_logits_final": torch.zeros(2, 4),
        "action_logits_visual": torch.zeros(2, 4),
        "dense_specificity_loss": torch.tensor(9.0),
        "action_specificity_loss": torch.tensor(2.0),
    }
    weights = {
        "action_final": 0.0,
        "action_visual": 0.0,
        "action_correction": 0.0,
        "action_two_way": 0.0,
        "action_soft_f1": 0.0,
        "action_cardinality": 0.0,
        "action_specificity": 0.5,
        "action_identity": 0.0,
        "action_near_boundary": 0.0,
        "action_anti_monopoly": 0.0,
    }
    result = meter_action_loss(output, torch.zeros(2, 4), weights)
    torch.testing.assert_close(result["total"], torch.tensor(1.0))
    torch.testing.assert_close(result["specificity"], torch.tensor(2.0))
    legacy = meter_action_loss({k: v for k, v in output.items() if k != "action_specificity_loss"}, torch.zeros(2, 4), weights)
    torch.testing.assert_close(legacy["specificity"], torch.zeros(()))


def test_schema_exposes_action_specific_factor_ownership() -> None:
    schema = default_meter_factor_schema()
    ownership = torch.tensor(schema.action_ownership)
    assert ownership.shape == (4, 21)
    # Follow-traffic is meaningful for longitudinal actions, not lateral ones.
    assert ownership[0, 1] > 0
    assert ownership[1, 1] > 0
    assert ownership[2, 1] == 0
    assert ownership[3, 1] == 0
    # Directional factors remain available only to their matching action.
    assert ownership[2, 12] > 0
    assert ownership[2, 18] == 0
    assert ownership[3, 18] > 0
    assert ownership[3, 12] == 0


def test_action_local_token_excludes_global_semantic_shortcut() -> None:
    head = TypedEvidenceStateHead(dim=8, factor_dim=21)
    anchor = torch.randn(2, 21, 8)
    state = torch.softmax(torch.randn(2, 21, 3), dim=-1)
    first = head.compose_action_token(anchor, state)
    second = head.compose_action_token(anchor, state)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 21, 8)


def test_action_specific_ownership_blocks_global_factor_monopoly() -> None:
    module = _transport()
    with torch.no_grad():
        module.action_factor_compatibility.zero_()
        module.action_factor_compatibility[:, 1] = 30.0
    ownership = torch.ones(4, 4)
    ownership[2:, 1] = 0.0
    output = module(
        torch.zeros(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.ones(1, 4),
        ownership,
        factor_source=torch.ones(1, 4),
        progress=1.0,
    )
    assert output["action_factor_weights"][0, :2, 1].gt(0).all()
    assert output["action_factor_weights"][0, 2:, 1].eq(0).all()


def test_exploration_floor_preserves_multiple_source_eligible_routes() -> None:
    module = _transport()
    with torch.no_grad():
        module.action_factor_compatibility.fill_(-30.0)
        module.action_factor_compatibility[:, 0] = 30.0
    output = module(
        torch.zeros(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.ones(1, 4),
        torch.ones(4, 4),
        factor_source=torch.ones(1, 4),
        progress=1.0,
    )
    weights = output["action_factor_weights"]
    assert weights[..., 1:].sum(-1).gt(0.02).all()
    assert (weights > 0).sum(-1).ge(2).all()


def test_pre_sparse_anti_monopoly_restores_gradient_to_alternative_route() -> None:
    module = _transport()
    with torch.no_grad():
        module.action_factor_compatibility.fill_(-12.0)
        module.action_factor_compatibility[:, 0] = 12.0
    output = module(
        torch.zeros(1, 4),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.ones(1, 4),
        torch.ones(4, 4),
        factor_source=torch.ones(1, 4),
        progress=1.0,
    )
    loss = meter_action_losses.action_transport_anti_monopoly_loss(
        output["action_factor_dense_weights"],
        output["action_factor_source_mask"],
    )
    loss.backward()
    assert loss > 0
    assert module.action_factor_compatibility.grad is not None
    assert module.action_factor_compatibility.grad[:, 1:].abs().sum() > 0
