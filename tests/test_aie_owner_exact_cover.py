from fate_oia.losses.aie_loss_registry import exact_owner_parameter_groups
from fate_oia.models.aie_oia_model import AIEOIAModel


def test_optimizer_owners_exactly_cover_trainable_parameters():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    groups = exact_owner_parameter_groups(model)
    assert set(groups) == {"primary", "action_evidence", "action_contribution", "reason_private"}

