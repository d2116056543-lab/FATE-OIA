import torch

from fate_oia.losses.precise_intervention_losses import incompatible_target_mask, matched_control_is_valid, packed_target_specific_interventions, target_specific_intervention_loss
from fate_oia.models.precise_oia_model import PRECISEOIAModel


def test_selected_effect_and_wrong_target_have_required_loss_direction():
    base = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    good = target_specific_intervention_loss(torch.full((2, 4), 0.5), torch.full((2, 4), 0.1), torch.full((2, 4), 0.1), base, base, target)["loss_intervention"]
    bad = target_specific_intervention_loss(torch.full((2, 4), 0.1), torch.full((2, 4), 0.5), torch.full((2, 4), 0.6), base, base, target)["loss_intervention"]
    assert good < bad


def test_matched_control_enforces_mass_and_nonoverlap():
    selected = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    control = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    valid = matched_control_is_valid(selected, control, torch.tensor([True]), torch.tensor([True]), torch.tensor([True]))
    assert valid.item() is True


def test_wrong_target_candidates_are_incompatible_with_selected_evidence_family():
    model = PRECISEOIAModel(use_mock_dino=True)
    traffic_light = next(index for index, row in enumerate(model.evidence_schema) if row["name"] == "traffic_light")
    fields = torch.tensor([traffic_light])
    action_mask = incompatible_target_mask(model, "action", fields)
    reason_mask = incompatible_target_mask(model, "reason", fields)
    assert action_mask.shape == (1, 4)
    assert reason_mask.shape == (1, 21)
    assert torch.equal(action_mask[0], ~model.exchange.family_mask_action[:, traffic_light])
    assert torch.equal(reason_mask[0], ~model.exchange.family_mask_reason[:, traffic_light])


def test_packed_intervention_reuses_cached_field_and_backpropagates():
    model = PRECISEOIAModel(use_mock_dino=True)
    output = model(torch.randn(2, 3, 360, 640))
    calls_before = model.dino.dino_call_count
    action = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    reason = torch.zeros(2, 21)
    reason[0, 3] = 1.0
    reason[1, 10] = 1.0
    losses = packed_target_specific_interventions(model, output, action, reason, max_pairs=4)
    losses["loss_intervention"].backward()
    assert model.dino.dino_call_count == calls_before
    assert 0 < losses["intervention_pair_count"].item() <= 4
    assert 0.0 <= losses["intervention_hard_rate"].item() <= 1.0
    assert 0.0 <= losses["intervention_easy_rate"].item() <= 1.0
    assert model.exchange.action_query.weight.grad is not None
