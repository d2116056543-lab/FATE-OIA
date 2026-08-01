import torch

from fate_oia.engine.eval_acpr_meter_oia import EXPENSIVE_SAME_FORWARD_MODES
from fate_oia.models.meter_oia_model import METEROIAModel


def test_targeted_schema_diagnostics_are_collected_on_expensive_epochs() -> None:
    for action in range(4):
        assert EXPENSIVE_SAME_FORWARD_MODES[f"schema_target_{action}"] == (
            f"schema_target_{action}",
        )


def test_targeted_schema_changes_only_the_requested_action_factor_route() -> None:
    torch.manual_seed(29)
    model = METEROIAModel(dim=384, use_mock_dino=True).eval()
    field = model.encode_images(torch.randn(2, 3, 360, 640))
    clean = model.decode_from_field(field, progress=1.0)
    targeted = model.decode_from_field(
        field, progress=1.0, diagnostic_modes=("schema_target_2",)
    )

    assert not torch.allclose(
        clean["action_factor_weights"][:, 2],
        targeted["action_factor_weights"][:, 2],
    )
    for action in (0, 1, 3):
        torch.testing.assert_close(
            clean["action_factor_weights"][:, action],
            targeted["action_factor_weights"][:, action],
        )
    torch.testing.assert_close(clean["reason_logits_final"], targeted["reason_logits_final"])
