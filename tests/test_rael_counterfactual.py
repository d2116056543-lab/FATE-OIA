from __future__ import annotations

import importlib
import inspect

import pytest
import torch
from torch import nn

from fate_oia.models.rael_relation_contributions import RAELPairwiseContribution


def _module():
    try:
        return importlib.import_module("fate_oia.losses.rael_counterfactual_losses")
    except ModuleNotFoundError as error:
        pytest.fail(f"P14 RED: counterfactual module is absent: {error}")


def _masks(*, batch: int = 1, height: int = 45, width: int = 80, device: str | torch.device = "cpu") -> torch.Tensor:
    masks = torch.zeros(batch, 20, height, width, device=device)
    # Selected slot 0 and valid control slot 1: same sector, equal four-patch
    # mass, nearby vertical centroid, and no overlap.
    masks[:, 0, 18:20, 10:12] = 1.0
    masks[:, 1, 18:20, 16:18] = 1.0
    # Deliberately invalid alternatives exercise sector, mass, overlap, and
    # vertical rejection in the deterministic control selector.
    masks[:, 2, 3:5, 30:32] = 1.0
    masks[:, 3, 18:20, 10:12] = 1.0
    masks[:, 4, 18:20, 22:25] = 1.0
    return masks


def _sectors(*, batch: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
    sectors = torch.zeros(batch, 20, 3, device=device)
    sectors[..., 0] = 1.0
    sectors[:, 2] = torch.tensor([0.0, 1.0, 0.0], device=device)
    return sectors


def _soft_mass_control_masks(*, candidate_pixels: int) -> torch.Tensor:
    masks = torch.zeros(1, 20, 45, 80)
    masks[:, 0, 18:22, 10:15] = 0.7
    masks[:, 1, 18:22, 30:35] = 0.7
    for index in range(20 - candidate_pixels):
        masks[:, 1, 18 + index // 5, 30 + index % 5] = 0.0
    return masks


def _hard_mass_control_masks(*, candidate_pixels: int) -> torch.Tensor:
    masks = torch.zeros(1, 20, 45, 80)
    masks[:, 0, 18:22, 10:15] = 1.0
    masks[:, 1, 18:22, 30:35] = 1.0
    for index in range(20 - candidate_pixels):
        masks[:, 1, 18 + index // 5, 30 + index % 5] = 0.0
    return masks


def _lexicographic_control_masks() -> torch.Tensor:
    masks = torch.zeros(1, 20, 45, 80)
    masks[:, 0, 18:22, 10:15] = 1.0
    # Slot 1 is valid but vertically farther than slot 2.  The vectorized
    # selector must rank the whole valid set rather than stop at slot 1.
    masks[:, 1, 20:24, 30:35] = 1.0
    masks[:, 2, 18:22, 50:55] = 1.0
    return masks


def _analytical(*, batch: int, targets: int, device: str | torch.device = "cpu") -> torch.Tensor:
    value = torch.zeros(batch, targets, 20, device=device)
    value[:, 0, 0] = 5.0
    value[:, 0, 1] = 1.0
    if targets > 1:
        value[:, 1, 0] = 3.0
    return value


class _OwnerReadout(nn.Module):
    def __init__(self, dim: int, targets: int) -> None:
        super().__init__()
        self.readout = nn.Linear(dim, targets, bias=False)
        self.contribution = nn.Linear(dim, targets, bias=False)
        nn.init.constant_(self.readout.weight, 0.4)
        nn.init.constant_(self.contribution.weight, -0.15)
        self.readout_calls = 0
        self.contribution_calls = 0

    def public_readout(self, field: torch.Tensor) -> torch.Tensor:
        self.readout_calls += 1
        pooled = field.mean(dim=(-1, -2)).to(dtype=self.readout.weight.dtype)
        return self.readout(pooled).float()

    def public_contribution(self, field: torch.Tensor) -> torch.Tensor:
        self.contribution_calls += 1
        pooled = field.mean(dim=(-1, -2)).to(dtype=self.contribution.weight.dtype)
        return self.contribution(pooled).float()


class _CountingDino:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, _: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("P14 must never invoke DINO")


@pytest.mark.parametrize("targets", [4, 21])
def test_analytical_deletion_equals_unary_plus_p10_incident_for_every_public_slot(targets: int) -> None:
    module = _module()
    unary = torch.randn(2, targets, 20, requires_grad=True)
    incident = torch.randn(2, targets, 20, requires_grad=True)
    output = module.analytical_deletion_deltas(
        unary_postgamma=unary,
        incident_pair_postgamma=incident,
    )
    assert output["deletion_delta"].shape == (2, targets, 20)
    assert torch.allclose(output["deletion_delta"], unary + incident, atol=1e-6)
    assert output["diagnostics"]["positive_deletion_mean"].grad_fn is None
    assert output["diagnostics"]["negative_deletion_mean"].grad_fn is None
    with pytest.raises(ValueError, match="20 public slots"):
        module.analytical_deletion_deltas(
            unary_postgamma=torch.zeros(1, targets, 21),
            incident_pair_postgamma=torch.zeros(1, targets, 21),
        )


@pytest.mark.parametrize("targets", [4, 21])
def test_analytical_deletion_matches_actual_p10_incident_interface_slot_by_slot(targets: int) -> None:
    module = _module()
    torch.manual_seed(1412 + targets)
    relation = RAELPairwiseContribution(num_targets=targets, dim=8)
    with torch.no_grad():
        relation.pair_output.normal_(std=0.2)
        relation.gamma_pair_raw.fill_(0.7)
    pair = relation(
        target_tokens=torch.randn(1, targets, 8),
        evidence_tokens=torch.randn(1, 20, 8),
        slot_masks=_masks(height=5, width=7),
        sector_probs=_sectors(),
        unary_public_pi=torch.softmax(torch.randn(1, targets, 20), dim=-1),
        reliability=torch.full((1, 20), 0.75),
    )
    unary = torch.randn(1, targets, 20)
    output = module.analytical_deletion_deltas(
        unary_postgamma=unary,
        incident_pair_postgamma=pair["incident_postgamma_by_slot"],
    )
    for slot in (0, 4, 11, 19):
        involved = (relation.pair_indices == slot).any(dim=-1)
        expected = unary[..., slot] + pair["pair_contributions"][..., involved].sum(dim=-1)
        assert torch.allclose(output["deletion_delta"][..., slot], expected, atol=1e-6)


def test_neighbor_replacement_is_out_of_place_same_dtype_and_only_changes_selected_region() -> None:
    module = _module()
    field = torch.arange(1 * 2 * 45 * 80, dtype=torch.float32).reshape(1, 2, 45, 80)
    source = field.clone()
    edge_mask = torch.zeros(1, 45, 80)
    edge_mask[:, :2, :2] = 1.0
    replacement, available = module.neighborhood_background_mean(field, edge_mask)
    assert available.item() is True
    modified = module.replace_region_with_neighbor_mean(field, edge_mask, replacement)
    assert modified.shape == field.shape and modified.dtype == field.dtype and modified.device == field.device
    assert torch.equal(field, source)
    support = edge_mask.bool().unsqueeze(1).expand_as(field)
    assert torch.equal(modified.masked_select(~support), field.masked_select(~support))
    assert not torch.equal(modified.masked_select(support), field.masked_select(support))
    empty = torch.zeros_like(edge_mask)
    full = torch.ones_like(edge_mask)
    assert module.neighborhood_background_mean(field, empty)[1].item() is False
    assert module.neighborhood_background_mean(field, full)[1].item() is False


def test_control_selection_is_deterministic_and_enforces_sector_vertical_mass_and_overlap() -> None:
    module = _module()
    masks = _masks()
    sectors = _sectors()
    first = module.select_equal_mass_control(
        slot_masks=masks,
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    second = module.select_equal_mass_control(
        slot_masks=masks.flip(0).flip(0),
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    assert bool(first["available"]) and bool(second["available"])
    assert isinstance(first["control_slot"], torch.Tensor)
    assert torch.equal(first["control_slot"], second["control_slot"]) and first["control_slot"].item() == 1
    assert bool(first["sector_match"])
    assert float(first["vertical_distance"]) <= 0.10
    assert abs(float(first["mass_ratio"]) - 1.0) <= 0.05
    assert float(first["overlap"]) < 0.05
    impossible = _masks()
    impossible[:, 1:] = 0.0
    unavailable = module.select_equal_mass_control(
        slot_masks=impossible,
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    assert not bool(unavailable["available"])
    assert isinstance(unavailable["control_slot"], torch.Tensor)
    assert unavailable["reason"] == "tensorized_availability"


@pytest.mark.parametrize("targets", [4, 21])
def test_feature_intervention_is_step_gated_and_replays_only_public_callbacks(targets: int) -> None:
    module = _module()
    torch.manual_seed(1407 + targets)
    field = torch.randn(1, 6, 45, 80, requires_grad=True)
    owner = _OwnerReadout(6, targets)
    base = owner.public_readout(field) + owner.public_contribution(field)
    owner.readout_calls = owner.contribution_calls = 0
    dino = _CountingDino()
    kwargs = dict(
        shared_field=field,
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=base,
        analytical_deletion=_analytical(batch=1, targets=targets),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["case-001.jpg"],
    )
    inactive = module.run_feature_intervention(optimizer_update=7, **kwargs)
    assert inactive["available"] is False
    assert inactive["effects"] is None and inactive["loss"] is None
    active = module.run_feature_intervention(optimizer_update=8, **kwargs)
    assert active["available"] is True
    assert active["case_id"] == "path:relative/case-001.jpg"
    assert active["selected_slot"].item() == 0 and active["control_slot"].item() == 1
    assert isinstance(active["control_slot"], torch.Tensor)
    assert owner.readout_calls == 2 and owner.contribution_calls == 2
    assert dino.calls == 0
    assert active["effects"]["d_selected"].requires_grad
    assert all(not value.requires_grad for value in active["diagnostics"].values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("targets", [4, 21])
def test_counterfactual_loss_has_exact_margin_sign_and_owner_only_gradient(targets: int) -> None:
    module = _module()
    expected = module.counterfactual_margin_loss(
        d_selected=torch.tensor(1.0),
        d_control=torch.tensor(4.0),
        d_target=torch.tensor(1.0),
        d_wrong=torch.tensor(3.0),
        margin=0.5,
    )
    assert torch.allclose(expected["loss"], torch.tensor(6.0))
    field = torch.randn(1, 5, 45, 80, requires_grad=True)
    owner = _OwnerReadout(5, targets)
    base = owner.public_readout(field) + owner.public_contribution(field)
    masks = _masks().requires_grad_()
    sectors = _sectors().requires_grad_()
    analytical = _analytical(batch=1, targets=targets).requires_grad_()
    output = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=field,
        slot_masks=masks,
        sector_probs=sectors,
        base_logits=base,
        analytical_deletion=analytical,
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["case-grad.jpg"],
    )
    assert output["available"] is True
    output["loss"].backward()
    assert owner.readout.weight.grad is not None and float(owner.readout.weight.grad.abs().sum()) > 0.0
    assert owner.contribution.weight.grad is not None and float(owner.contribution.weight.grad.abs().sum()) > 0.0
    assert masks.grad is None and sectors.grad is None and analytical.grad is None
    assert output["selection"]["selected_slot"].requires_grad is False
    assert output["selection"]["control_slot"].requires_grad is False
    for name in ("selected_effect", "control_effect", "target_effect", "wrong_effect", "positive_analytical_effect", "negative_analytical_effect"):
        assert output["diagnostics"][name].grad_fn is None


@pytest.mark.parametrize("targets", [4, 21])
def test_counterfactual_detaches_base_owner_and_shared_field_but_updates_replay_owner(targets: int) -> None:
    module = _module()
    torch.manual_seed(1416 + targets)
    field = torch.randn(1, 5, 45, 80, requires_grad=True)
    base_owner = _OwnerReadout(5, targets)
    replay_owner = _OwnerReadout(5, targets)
    base_logits = base_owner.public_readout(field) + base_owner.public_contribution(field)
    masks = _masks().requires_grad_()
    sectors = _sectors().requires_grad_()
    analytical = _analytical(batch=1, targets=targets).requires_grad_()

    output = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=field,
        slot_masks=masks,
        sector_probs=sectors,
        base_logits=base_logits,
        analytical_deletion=analytical,
        public_readout=replay_owner.public_readout,
        public_contribution=replay_owner.public_contribution,
        case_ids=["case-base-isolation.jpg"],
    )

    assert output["available"] is True
    output["loss"].backward()
    assert base_owner.readout.weight.grad is None
    assert base_owner.contribution.weight.grad is None
    assert field.grad is None
    assert replay_owner.readout.weight.grad is not None
    assert float(replay_owner.readout.weight.grad.abs().sum()) > 0.0
    assert replay_owner.contribution.weight.grad is not None
    assert float(replay_owner.contribution.weight.grad.abs().sum()) > 0.0
    assert masks.grad is None and sectors.grad is None and analytical.grad is None


def test_neighborhood_mean_is_finite_at_float32_extremes_and_active_path_rejects_nonfinite_inputs() -> None:
    module = _module()
    extreme_field = torch.full((1, 3, 45, 80), 1.0e38)
    region = torch.zeros(1, 45, 80)
    region[:, 20:22, 10:12] = 1.0
    mean, available = module.neighborhood_background_mean(extreme_field, region)
    assert available.item() is True and torch.isfinite(mean).all()

    field = torch.randn(1, 3, 45, 80)
    owner = _OwnerReadout(3, 4)
    kwargs = dict(
        optimizer_update=8,
        shared_field=field,
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=owner.public_readout(field) + owner.public_contribution(field),
        analytical_deletion=_analytical(batch=1, targets=4),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["finite-check.jpg"],
    )
    for name, nonfinite in (
        ("shared_field", torch.full_like(field, float("nan"))),
        ("slot_masks", torch.full_like(_masks(), float("inf"))),
        ("sector_probs", torch.full_like(_sectors(), float("nan"))),
        ("base_logits", torch.full((1, 4), float("inf"))),
        ("analytical_deletion", torch.full((1, 4, 20), float("nan"))),
    ):
        bad = dict(kwargs)
        bad[name] = nonfinite
        with pytest.raises((ValueError, RuntimeError), match="finite"):
            module.run_feature_intervention(**bad)
    with pytest.raises((ValueError, RuntimeError), match="public_readout.*finite"):
        module.run_feature_intervention(
            **{
                **kwargs,
                "public_readout": lambda _: torch.full((1, 4), float("nan")),
            }
        )
    with pytest.raises(ValueError, match="unary_postgamma.*finite"):
        module.analytical_deletion_deltas(
            unary_postgamma=torch.full((1, 4, 20), float("inf")),
            incident_pair_postgamma=torch.zeros(1, 4, 20),
        )


def test_control_selection_uses_soft_mass_scale_tolerance_and_lexicographic_best_valid_slot() -> None:
    module = _module()
    sectors = _sectors()
    exact_boundary = module.select_equal_mass_control(
        slot_masks=_hard_mass_control_masks(candidate_pixels=19),
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    assert bool(exact_boundary["available"]) and exact_boundary["control_slot"].item() == 1
    assert float(exact_boundary["mass_ratio"]) == pytest.approx(0.95)
    accepted = module.select_equal_mass_control(
        slot_masks=_soft_mass_control_masks(candidate_pixels=19),
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    assert bool(accepted["available"]) and accepted["control_slot"].item() == 1
    assert float(accepted["selected_mass"]) == pytest.approx(14.0)
    assert float(accepted["mass_ratio"]) == pytest.approx(0.95, abs=1.0e-6)
    rejected = module.select_equal_mass_control(
        slot_masks=_soft_mass_control_masks(candidate_pixels=18),
        sector_probs=sectors,
        sample_index=0,
        selected_slot=0,
    )
    assert not bool(rejected["available"])

    best = module.select_equal_mass_control(
        slot_masks=_lexicographic_control_masks(),
        sector_probs=torch.nn.functional.one_hot(torch.zeros(1, 20, dtype=torch.long), num_classes=3).float(),
        sample_index=0,
        selected_slot=0,
    )
    assert bool(best["available"]) and best["control_slot"].item() == 2
    assert "for candidate in range" not in inspect.getsource(module.select_equal_mass_control)


def test_optimizer_update_case_identity_and_diagnostic_schema_are_strict() -> None:
    module = _module()
    field = torch.randn(1, 4, 45, 80)
    owner = _OwnerReadout(4, 4)
    kwargs = dict(
        shared_field=field,
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=owner.public_readout(field) + owner.public_contribution(field),
        analytical_deletion=_analytical(batch=1, targets=4),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=[r"C:\BDD\seq\..\A.JPG."],
    )
    for invalid_update in (True, 0, 8.0, 8.5, -1):
        with pytest.raises(ValueError, match="optimizer_update"):
            module.run_feature_intervention(optimizer_update=invalid_update, **kwargs)
    for invalid_interval in (True, 0, 8.0, 8.5, -1):
        with pytest.raises(ValueError, match="every_optimizer_updates"):
            module.run_feature_intervention(
                optimizer_update=8,
                every_optimizer_updates=invalid_interval,
                **kwargs,
            )
    inactive = module.run_feature_intervention(optimizer_update=7, **kwargs)
    active = module.run_feature_intervention(optimizer_update=8, **kwargs)
    assert active["case_id"] == "path:drive:c:/bdd/a.jpg"
    assert inactive["effects"] is None and inactive["loss"] is None
    assert set(inactive["diagnostics"]) == set(active["diagnostics"])
    for name, inactive_value in inactive["diagnostics"].items():
        active_value = active["diagnostics"][name]
        assert isinstance(inactive_value, torch.Tensor) and isinstance(active_value, torch.Tensor)
        assert inactive_value.device == active_value.device == field.device
        assert inactive_value.dtype == active_value.dtype
        assert inactive_value.shape == active_value.shape
    assert inactive["diagnostics"]["computed"].item() is False
    assert active["diagnostics"]["computed"].item() is True
    assert torch.isnan(inactive["diagnostics"]["selected_effect"])
    for unsafe_ids, error_type in (({"case.jpg": "unexpected"}, TypeError), ([{"case": "unsafe"}], TypeError), ([r"C:\..\escape.jpg"], ValueError)):
        with pytest.raises(error_type):
            module.run_feature_intervention(optimizer_update=8, **{**kwargs, "case_ids": unsafe_ids})


def test_scheduled_control_routing_keeps_candidate_indices_device_resident_until_one_final_sync() -> None:
    module = _module()
    source = inspect.getsource(module.run_feature_intervention)
    control_source = inspect.getsource(module.select_equal_mass_control)
    assert source.count(".item(") == 1
    assert control_source.count(".item(") == 0

    field = torch.randn(1, 4, 45, 80)
    owner = _OwnerReadout(4, 4)
    output = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=field,
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=owner.public_readout(field) + owner.public_contribution(field),
        analytical_deletion=_analytical(batch=1, targets=4),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["tensor-control.jpg"],
    )
    assert output["available"] is True
    assert all(
        isinstance(output[name], torch.Tensor) and output[name].ndim == 0
        for name in ("sample_index", "target_index", "wrong_target_index", "selected_slot", "control_slot")
    )


def test_p14_api_rejects_image_dino_cache_and_supports_noncontiguous_batch_one_extremes() -> None:
    module = _module()
    signature = inspect.signature(module.run_feature_intervention)
    forbidden = {"image", "images", "dino", "extractor", "cache"}
    assert not forbidden.intersection(signature.parameters)
    field = torch.randn(1, 4, 45, 160)[..., ::2].requires_grad_()
    assert not field.is_contiguous()
    owner = _OwnerReadout(4, 4)
    output = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=field,
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=owner.public_readout(field) + owner.public_contribution(field),
        analytical_deletion=_analytical(batch=1, targets=4),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["case-nc.jpg"],
    )
    assert output["available"] is True
    assert torch.isfinite(output["loss"])
    extreme = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=torch.full_like(field, 1e20),
        slot_masks=_masks(),
        sector_probs=_sectors(),
        base_logits=owner.public_readout(torch.full_like(field, 1e20)) + owner.public_contribution(torch.full_like(field, 1e20)),
        analytical_deletion=_analytical(batch=1, targets=4),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["case-extreme.jpg"],
    )
    assert extreme["available"] is True and torch.isfinite(extreme["loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="P14 CUDA BF16 probe requires CUDA")
@pytest.mark.parametrize("targets", [4, 21])
def test_p14_cuda_bf16_k4_k21_is_finite(targets: int) -> None:
    module = _module()
    device = torch.device("cuda")
    field = torch.randn(1, 4, 45, 160, device=device, dtype=torch.bfloat16)[..., ::2].requires_grad_()
    owner = _OwnerReadout(4, targets).to(device=device, dtype=torch.bfloat16)
    output = module.run_feature_intervention(
        optimizer_update=8,
        shared_field=field,
        slot_masks=_masks(device=device).to(torch.bfloat16),
        sector_probs=_sectors(device=device).to(torch.bfloat16),
        base_logits=owner.public_readout(field) + owner.public_contribution(field),
        analytical_deletion=_analytical(batch=1, targets=targets, device=device).to(torch.bfloat16),
        public_readout=owner.public_readout,
        public_contribution=owner.public_contribution,
        case_ids=["case-cuda.jpg"],
    )
    assert output["available"] is True and torch.isfinite(output["loss"])
    output["loss"].backward()
    assert owner.readout.weight.grad is not None and torch.isfinite(owner.readout.weight.grad).all()
