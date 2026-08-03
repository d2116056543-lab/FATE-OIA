import pytest
import torch
from torch import nn

from fate_oia.losses.save_loss_registry import (
    ACTION_PREDICATE_BRIDGE_SCALE,
    SAVE_FIREWALL_DETACHES,
    SAVE_GRADIENT_OWNER_ALLOWLIST,
    SAVE_LOSS_REGISTRATION_OWNER_MAP,
    SAVE_PARAMETER_OWNER_GROUPS,
    SAVELossRegistry,
    build_save_optimizer_groups,
    fixed_action_predicate_bridge,
    validate_optimizer_groups,
    validate_save_firewall_contract,
    validate_save_parameter_ownership,
)
from fate_oia.models.save_oia_model import SAVEOIAModel


EXPECTED_REGISTRATION_OWNERS = {
    "action_final": "action_multi_inquiry",
    "action_base": "foundation_joint",
    "action_evidence_aux": "action_multi_inquiry",
    "action_utility_cf": "utility_bridge",
    "action_utility_dense": "utility_bridge",
    "action_sufficiency": "action_multi_inquiry",
    "action_necessity": "action_multi_inquiry",
    "action_control": "action_multi_inquiry",
    "action_preserve": "action_multi_inquiry",
    "action_soft_f1": "foundation_joint",
    "action_cardinality": "foundation_joint",
    "action_easy": "foundation_joint",
    "reason_benchmark": "private_reason",
    "reason_private_direct": "private_reason",
    "reason_clean": "clean_reason_adapter",
    "reason_rank": "private_reason",
    "reason_soft_f1": "private_reason",
    "reason_bbam": "private_reason",
    "reason_view_consistency": "private_reason",
    "reason_pu_private": "private_reason",
    "measurement_anchor": "predicate_measurement",
    "measurement_state": "predicate_measurement",
    "measurement_null": "predicate_measurement",
    "measurement_matched_background": "predicate_measurement",
    "measurement_mirror": "predicate_measurement",
    "measurement_identity": "predicate_measurement",
}

ACTION_FINAL_OWNERS = frozenset(
    {
        "foundation_joint",
        "predicate_measurement",
        "action_multi_inquiry",
        "utility_bridge",
        "clean_reason_adapter",
    }
)
ACTION_MECHANISM_OWNERS = frozenset(
    {"predicate_measurement", "action_multi_inquiry", "utility_bridge"}
)
ACTION_REGULARIZER_OWNERS = frozenset(
    {"foundation_joint", "action_multi_inquiry"}
)
PRIVATE_REASON_OWNERS = frozenset({"private_reason"})
PREDICATE_OWNERS = frozenset({"predicate_measurement"})

EXPECTED_GRADIENT_OWNERS = {
    "action_final": ACTION_FINAL_OWNERS,
    "action_base": frozenset({"foundation_joint"}),
    "action_evidence_aux": ACTION_MECHANISM_OWNERS,
    "action_utility_cf": ACTION_MECHANISM_OWNERS,
    "action_utility_dense": ACTION_MECHANISM_OWNERS,
    "action_sufficiency": ACTION_MECHANISM_OWNERS,
    "action_necessity": ACTION_MECHANISM_OWNERS,
    "action_control": ACTION_MECHANISM_OWNERS,
    "action_preserve": ACTION_MECHANISM_OWNERS,
    "action_soft_f1": ACTION_REGULARIZER_OWNERS,
    "action_cardinality": ACTION_REGULARIZER_OWNERS,
    "action_easy": ACTION_REGULARIZER_OWNERS,
    "reason_benchmark": PRIVATE_REASON_OWNERS,
    "reason_private_direct": PRIVATE_REASON_OWNERS,
    "reason_clean": frozenset(
        {"foundation_joint", "predicate_measurement", "clean_reason_adapter"}
    ),
    "reason_rank": PRIVATE_REASON_OWNERS,
    "reason_soft_f1": PRIVATE_REASON_OWNERS,
    "reason_bbam": PRIVATE_REASON_OWNERS,
    "reason_view_consistency": PRIVATE_REASON_OWNERS,
    "reason_pu_private": PRIVATE_REASON_OWNERS,
    "measurement_anchor": PREDICATE_OWNERS,
    "measurement_state": PREDICATE_OWNERS,
    "measurement_null": PREDICATE_OWNERS,
    "measurement_matched_background": PREDICATE_OWNERS,
    "measurement_mirror": PREDICATE_OWNERS,
    "measurement_identity": PREDICATE_OWNERS,
}


class _OwnedSAVEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foundation_joint = nn.Linear(2, 2)
        self.predicate_measurement = nn.Linear(2, 2)
        self.action_multi_inquiry = nn.Linear(2, 2)
        self.utility_bridge = nn.Linear(2, 2)
        self.clean_reason_adapter = nn.Linear(2, 2)
        self.private_reason = nn.Linear(2, 2)
        self.dino = nn.Linear(2, 2)
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)


def test_save_optimizer_owner_sets_exactly_cover_trainable_non_dino_parameters() -> None:
    model = _OwnedSAVEModel()

    report = validate_save_parameter_ownership(model)
    groups = build_save_optimizer_groups(model)
    grouped = validate_optimizer_groups(groups, model)

    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("dino.")
    }
    actual = set().union(*(set(names) for names in report.owner_sets.values()))
    assert actual == expected
    assert set(report.owner_sets) == set(SAVE_PARAMETER_OWNER_GROUPS)
    assert grouped.owner_sets == report.owner_sets
    assert len(groups) == 6
    assert len({id(parameter) for group in groups for parameter in group["params"]}) == len(expected)


def test_save_optimizer_validation_rejects_overlapping_prefix_ownership() -> None:
    parameter = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="DUPLICATE_OWNER"):
        validate_save_parameter_ownership(
            {"foundation_joint.shared": parameter},
            owner_prefixes={
                "foundation_joint": ("foundation_joint.",),
                "predicate_measurement": ("foundation_joint.",),
            },
        )


def test_save_optimizer_validation_rejects_an_unowned_trainable_parameter() -> None:
    model = nn.Module()
    model.unknown_representation = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="UNOWNED_PARAMETER"):
        validate_save_parameter_ownership(model)


def test_save_optimizer_validation_rejects_duplicate_and_missing_group_membership() -> None:
    model = _OwnedSAVEModel()
    groups = build_save_optimizer_groups(model)
    groups[1]["params"].append(groups[0]["params"][0])

    with pytest.raises(ValueError, match="DUPLICATE_OWNER"):
        validate_optimizer_groups(groups, model)


def test_save_optimizer_rejects_trainable_posthoc_threshold_or_temperature() -> None:
    model = _OwnedSAVEModel()
    model.threshold = nn.Parameter(torch.zeros(1))
    model.temperature = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="POSTHOC_PARAMETER_INCLUDED"):
        build_save_optimizer_groups(model)


def test_save_predicate_measurement_temperature_is_not_misclassified_as_posthoc() -> None:
    """The frozen CalAlign predicate measurement temperature is not a deploy calibrator."""
    model = SAVEOIAModel(use_mock_dino=True)
    groups = build_save_optimizer_groups(model)
    grouped_names = {
        group["group_name"]: {id(parameter) for parameter in group["params"]}
        for group in groups
    }
    parameter = dict(model.named_parameters())["foundation.predicate_head.temperature"]
    assert id(parameter) in grouped_names["predicate_measurement"]


def test_save_loss_registration_and_gradient_owner_tables_are_exact() -> None:
    registry = SAVELossRegistry()

    assert SAVE_LOSS_REGISTRATION_OWNER_MAP == EXPECTED_REGISTRATION_OWNERS
    assert SAVE_GRADIENT_OWNER_ALLOWLIST == EXPECTED_GRADIENT_OWNERS
    assert registry.loss_owner_map() == EXPECTED_REGISTRATION_OWNERS
    assert registry.gradient_owner_allowlist() == EXPECTED_GRADIENT_OWNERS
    assert set(EXPECTED_REGISTRATION_OWNERS.values()) == set(SAVE_PARAMETER_OWNER_GROUPS)


def test_save_registration_owner_may_be_an_allowed_gradient_owner() -> None:
    registry = SAVELossRegistry(expected_terms=("action_final",))

    registry.add(
        "action_final",
        torch.tensor(1.0),
        owner="action_multi_inquiry",
        gradient_owners=ACTION_FINAL_OWNERS,
    )
    registry.validate_complete()


def test_save_registry_rejects_nonapproved_owner_table_overrides() -> None:
    with pytest.raises(ValueError, match="OWNER_MISMATCH"):
        SAVELossRegistry(
            expected_terms=("action_final",),
            registration_owners={"action_final": "foundation_joint"},
        )
    with pytest.raises(ValueError, match="OWNER_MISMATCH"):
        SAVELossRegistry(
            expected_terms=("action_final",),
            gradient_owner_allowlist={
                "action_final": {"foundation_joint", "action_multi_inquiry"}
            },
        )


def test_save_bridge_and_firewall_contracts_are_fixed_and_fail_closed() -> None:
    assert ACTION_PREDICATE_BRIDGE_SCALE == 0.05
    value = torch.tensor([2.0], requires_grad=True)
    fixed_action_predicate_bridge(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([0.05]))

    validated = validate_save_firewall_contract(SAVE_FIREWALL_DETACHES)
    assert validated["predicate_action_bridge_scale"] == 0.05
    assert set(validated["required_detaches"]) == set(SAVE_FIREWALL_DETACHES)

    with pytest.raises(ValueError, match="FIREWALL_DETACH_MISSING"):
        validate_save_firewall_contract({"grounding_to_foundation": False})
