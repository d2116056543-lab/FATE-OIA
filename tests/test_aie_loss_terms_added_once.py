import pytest
import torch

from fate_oia.losses.aie_loss_registry import AIELossRegistry


def test_registry_rejects_duplicate_and_missing_terms():
    registry = AIELossRegistry({"a": 1.0, "b": 2.0}); registry.add("a", "primary", torch.tensor(1.0))
    with pytest.raises(ValueError): registry.add("a", "primary", torch.tensor(1.0))
    with pytest.raises(ValueError): registry.total()

