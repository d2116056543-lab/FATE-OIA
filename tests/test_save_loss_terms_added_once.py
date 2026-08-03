import pytest
import torch

from fate_oia.losses.save_loss_registry import (
    SAVE_LOSS_TERM_NAMES,
    SAVE_LOSS_WEIGHTS,
    SAVELossRegistry,
    SAVELossRuntime,
    build_save_loss_registry,
)


def _complete_registry() -> SAVELossRegistry:
    registry = SAVELossRegistry()
    for name in SAVE_LOSS_TERM_NAMES:
        registry.add(name, torch.tensor(1.0))
    return registry


def test_save_loss_registry_reconstructs_the_plan_once_with_runtime_counts() -> None:
    registry = _complete_registry()

    assert set(registry.runtime_call_counts()) == set(SAVE_LOSS_TERM_NAMES)
    assert all(count == 1 for count in registry.runtime_call_counts().values())
    assert sum(float(value) for value in registry.weighted_values().values()) == pytest.approx(
        float(sum(SAVE_LOSS_WEIGHTS.values()))
    )
    assert all(row["call_count"] == 1 for row in registry.artifact())


def test_save_loss_registry_rejects_duplicate_terms() -> None:
    registry = SAVELossRegistry()
    registry.add("action_final", torch.tensor(1.0))

    with pytest.raises(ValueError, match="LOSS_DUPLICATED"):
        registry.add("action_final", torch.tensor(1.0))


def test_save_loss_registry_rejects_missing_terms_before_total() -> None:
    registry = SAVELossRegistry()
    for name in SAVE_LOSS_TERM_NAMES[:-1]:
        registry.add(name, torch.tensor(1.0))

    with pytest.raises(ValueError, match="LOSS_MISSING"):
        registry.validate_complete()


def test_save_loss_registry_rejects_preweighted_values_and_noncanonical_weights() -> None:
    registry = SAVELossRegistry()

    with pytest.raises(ValueError, match="LOSS_DOUBLE_WEIGHTED"):
        registry.add("action_final", torch.tensor(1.0), weighted=True)
    with pytest.raises(ValueError, match="LOSS_DOUBLE_WEIGHTED"):
        registry.add("action_base", torch.tensor(1.0), weight=0.70)


def test_save_loss_runtime_rejects_a_second_invocation() -> None:
    runtime = SAVELossRuntime()
    runtime.invoke("action_final", lambda: torch.tensor(1.0))

    with pytest.raises(ValueError, match="LOSS_DUPLICATED"):
        runtime.invoke("action_final", lambda: torch.tensor(1.0))


def test_save_loss_registry_exposes_a_single_backward_contract() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    registry = SAVELossRegistry()
    for name in SAVE_LOSS_TERM_NAMES:
        registry.add(name, parameter * 0.0 + 1.0)

    registry.backward()
    assert parameter.grad is not None

    with pytest.raises(ValueError, match="BACKWARD_DUPLICATED"):
        registry.backward()


def test_bundle_totals_are_skipped_and_raw_terms_are_registered_once() -> None:
    action = {name.removeprefix("action_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("action_")}
    reason = {name.removeprefix("reason_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("reason_")}
    measurement = {name.removeprefix("measurement_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("measurement_")}
    action["total"] = torch.tensor(999.0); reason["total"] = torch.tensor(999.0); measurement["total"] = torch.tensor(999.0)
    registry = build_save_loss_registry(action=action, reason=reason, measurement=measurement)
    assert all(count == 1 for count in registry.runtime_call_counts().values())


def test_grounding_diagnostics_do_not_double_register_plan_level_terms() -> None:
    action = {name.removeprefix("action_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("action_")}
    reason = {name.removeprefix("reason_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("reason_")}
    measurement = {name.removeprefix("measurement_"): torch.tensor(1.0) for name in SAVE_LOSS_TERM_NAMES if name.startswith("measurement_")}
    measurement.update({"anchor_nll": torch.tensor(1.0), "anchor_dice": torch.tensor(1.0), "observability": torch.tensor(1.0), "discrimination": torch.tensor(1.0), "ontology_identity": torch.tensor(1.0)})
    registry = build_save_loss_registry(action=action, reason=reason, measurement=measurement)
    assert registry.runtime_call_counts()["measurement_anchor"] == 1
    assert registry.runtime_call_counts()["measurement_null"] == 1
