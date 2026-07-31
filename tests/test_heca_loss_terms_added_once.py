import torch

from fate_oia.optim.heca_optimization import HECALossRegistry


def test_loss_registry_rejects_duplicates_and_reconstructs_total() -> None:
    registry = HECALossRegistry()
    registry.add("action_final", torch.tensor(2.0), 1.0, owner="action")
    registry.add("reason_final", torch.tensor(3.0), 0.5, owner="reason")
    torch.testing.assert_close(registry.total(), torch.tensor(3.5))
    try:
        registry.add("action_final", torch.tensor(1.0), 1.0, owner="action")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate loss term was accepted")

