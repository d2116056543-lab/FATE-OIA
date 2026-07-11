from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _model() -> MOSAICADModel:
    return MOSAICADModel(
        config_root=CONFIG_ROOT,
        use_mock_dino=True,
        highres_topk=16,
        midres_topk=8,
        decoder_layers=1,
        self_attention_heads=4,
    )


def test_formal_model_is_independent_direct_image_and_returns_all_raw_branches() -> None:
    model = _model().eval()
    output = model(torch.randn(1, 3, 360, 640), return_masks=True)
    assert output["action_logits_visual"].shape == (1, 4)
    assert output["action_logits_state"].shape == (1, 4)
    assert output["action_state_gate"].shape == (1, 4)
    assert output["action_logits_raw"].shape == (1, 4)
    assert output["reason_logits_latent"].shape == (1, 21)
    assert output["factor_presence_logits"].shape == (1, 24)
    assert output["decision_state_prob"].shape == (1, 8)
    assert output["factor_soft_masks"].shape == (1, 24, 45, 80)
    assert output["sampling_coordinates"].shape == (1, 24, 2, 4, 12, 2)


@pytest.mark.parametrize("prior_mode", ["full", "content_only", "prior_only"])
def test_full_model_audit_modes_preserve_batch_and_final_output_shapes(prior_mode: str) -> None:
    output = _model().eval()(torch.randn(2, 3, 360, 640), prior_mode=prior_mode)
    assert output["factor_presence_prob"].shape == (2, 24)
    assert output["decision_state_prob"].shape == (2, 8)
    assert output["action_logits_raw"].shape == (2, 4)
    assert output["reason_logits_latent"].shape == (2, 21)


def test_phase_a_zero_controls_recover_action_visual_branch_exactly() -> None:
    model = _model().eval()
    model.set_phase_controls(state_residual_scale=0.0, action_state_gate_cap=0.0)
    output = model(torch.randn(1, 3, 360, 640))
    assert torch.count_nonzero(output["decision_state_residual"]) == 0
    assert torch.count_nonzero(output["action_state_gate"]) == 0
    assert torch.equal(output["action_logits_raw"], output["action_logits_visual"])


def test_reason_decoder_mutation_cannot_change_action_logits() -> None:
    torch.manual_seed(53)
    model = _model().eval()
    images = torch.randn(1, 3, 360, 640)
    before = model(images)["action_logits_raw"]
    with torch.no_grad():
        for parameter in model.reason_decoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
    after = model(images)["action_logits_raw"]
    assert torch.equal(before, after)


def test_forward_signature_physically_excludes_labels_geometry_and_training_metadata() -> None:
    signature = inspect.signature(MOSAICADModel.forward)
    assert list(signature.parameters) == ["self", "images", "prior_mode", "return_masks"]
    model = _model()
    with pytest.raises(TypeError):
        model(torch.randn(1, 3, 360, 640), reason_labels=torch.zeros(1, 21))
    with pytest.raises(TypeError):
        model(torch.randn(1, 3, 360, 640), geometry_records=[{}])


def test_dino_is_frozen_no_grad_and_old_acpr_model_is_not_instantiated() -> None:
    model = _model()
    assert all(not parameter.requires_grad for parameter in model.dino.parameters())
    output = model.dino(torch.randn(1, 3, 360, 640))
    assert output["patch_tokens_by_layer"].requires_grad is False
    module_names = {module.__class__.__name__ for module in model.modules()}
    assert "ACPROIAModel" not in module_names
    assert "ACPRLabelTrunk" not in module_names


def test_return_masks_false_drops_only_large_audit_tensors() -> None:
    model = _model().eval()
    output = model(torch.randn(1, 3, 360, 640), return_masks=False)
    assert "factor_soft_masks" not in output
    assert "sampling_coordinates" not in output
    assert output["action_logits_raw"].shape == (1, 4)
    assert output["reason_logits_latent"].shape == (1, 21)


def test_phase_controls_round_trip_checkpoint_without_forward_item_sync() -> None:
    model = _model()
    model.set_phase_controls(
        state_residual_scale=0.10,
        action_state_gate_cap=0.20,
        reason_state_contribution_cap=0.15,
    )
    restored = _model()
    restored.load_state_dict(model.state_dict())
    assert restored._state_residual_scale_value == pytest.approx(0.10)
    assert restored._action_state_gate_cap_value == pytest.approx(0.20)
    assert restored._reason_state_contribution_cap_value == pytest.approx(0.15)
    forward_source = inspect.getsource(MOSAICADModel.forward)
    assert ".item()" not in forward_source
