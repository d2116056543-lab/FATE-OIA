import pytest
import torch

from fate_oia.losses.pact_loss_registry import PACTLossRegistry, exact_pact_owner_groups
from fate_oia.models.pact_oia_model import PACTOIAModel


def test_loss_registry_rejects_duplicates_and_missing_terms():
    registry = PACTLossRegistry({"a": 1.0, "b": 2.0})
    registry.add("a", "context", torch.ones(()))
    with pytest.raises(ValueError):
        registry.add("a", "context", torch.ones(()))
    with pytest.raises(ValueError):
        registry.total()


def test_optimizer_ownership_is_exact_and_disjoint():
    model = PACTOIAModel(use_mock_dino=True)
    groups = exact_pact_owner_groups(model)
    ids = [id(parameter) for parameters in groups.values() for parameter in parameters]
    assert len(ids) == len(set(ids))
    assert {id(p) for p in model.parameters() if p.requires_grad} == set(ids)
