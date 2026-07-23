from pathlib import Path

import torch
import yaml

from fate_oia.engine.precise_curriculum import curriculum_state_for_epoch
from fate_oia.models.precise_oia_model import PRECISEOIAModel


ROOT = Path(__file__).resolve().parents[1]


def test_model_constructor_wires_yaml_values_instead_of_relying_on_matching_defaults():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    config["visual"]["adapter_hidden"] = 96
    config["category"]["heads"] = 2
    config["evidence"]["latent_slots"] = 3
    config["reason"]["annotation_rank"] = 4
    config["reason"]["semantic_weight_floor"] = 0.30
    config["exchange"]["overlap_tau"] = 0.12
    config["exchange"]["reliability_eps"] = 0.002
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True, model_config=config)
    assert model.visual_field.action_foundation[0].down.out_features == 96
    assert model.category_decoder.action_cross.num_heads == 2
    assert model.evidence_fields.latent_slots == 3
    assert model.annotation_head.down[0].out_features == 4
    assert model.exchange.overlap_tau == 0.12
    assert model.exchange.reliability_eps == 0.002
    assert model.semantic_weight_floor == 0.30


def test_full_model_forward_exposes_all_formal_branches_without_structured_input():
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True)
    output = model(torch.randn(1, 3, 360, 640))
    required = {
        "action_logits_direct", "action_logits_reread", "action_logits_exchange_ungated", "action_logits_exchange_certified", "action_logits_final_raw",
        "reason_logits_direct", "reason_logits_semantic", "reason_logits_observed", "reason_logits_final_raw", "action_logits_deploy", "reason_logits_deploy",
        "explicit_evidence_tokens", "latent_evidence_tokens", "evidence_reliability", "exchange_overlap", "exchange_gate", "action_exchange_delta", "reason_exchange_delta", "reference_points", "annotation_delta", "branch_logits", "diagnostics"
    }
    assert required <= set(output)
    assert output["action_logits_final_raw"].shape == (1, 4)
    assert output["reason_logits_final_raw"].shape == (1, 21)
    assert output["exchange_overlap"].shape == (1, 4, 21)
    assert output["reference_points"].shape == (1, 25, 3, 4, 2)
    assert output["reason_latent_delta"].shape == (1, 21, 384)
    assert output["reason_latent_attention"].shape == (1, 21, 6)
    for diagnostic in (
        "action_reason_message_norm", "reason_action_message_norm",
        "evidence_attention_entropy", "evidence_effective_support",
        "reason_token_shuffle_delta", "evidence_shuffle_delta",
    ):
        assert diagnostic in output
        assert torch.isfinite(output[diagnostic]).all()


def test_final_action_is_not_observed_reason_path_and_observed_loss_is_firewalled():
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True)
    output = model(torch.randn(1, 3, 360, 640))
    output["reason_logits_observed"].square().mean().backward()
    action_parameters = model.owned_parameters()["action_foundation"] + model.owned_parameters()["action_decoder"]
    evidence_parameters = model.owned_parameters()["evidence_core"]
    assert all(parameter.grad is None for parameter in action_parameters)
    assert all(parameter.grad is None for parameter in evidence_parameters)
    assert any(parameter.grad is not None for parameter in model.owned_parameters()["annotation_adapter"])


def test_foundation_stage_uses_refined_direct_anchor_and_zero_effective_deltas():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True, model_config=config).eval()
    model.set_curriculum_state(curriculum_state_for_epoch(config, 0))
    with torch.no_grad():
        output = model(torch.randn(1, 3, 360, 640))
    assert torch.equal(output["action_logits_final_raw"], output["action_logits_refined_direct_anchor"])
    assert torch.equal(output["reason_logits_semantic"], output["reason_logits_refined_direct_anchor"])
    assert torch.equal(output["reason_logits_final_raw"], output["reason_logits_refined_direct_anchor"])
    for key in (
        "action_reread_delta_effective",
        "reason_reread_delta_effective",
        "action_exchange_delta_effective",
        "reason_exchange_delta_effective",
        "reason_latent_delta_effective",
        "annotation_delta_effective",
        "threshold_logit_effective",
    ):
        assert torch.count_nonzero(output[key]) == 0


def test_partial_curriculum_scales_each_boundary_once_and_scales_deploy_theta():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    model = PRECISEOIAModel(ROOT / "configs", use_mock_dino=True, model_config=config).eval()
    with torch.no_grad():
        model.threshold_head.theta_group.fill_(0.4)
        model.threshold_head.theta_delta.zero_()
    state = curriculum_state_for_epoch(config, 2)
    model.set_curriculum_state(state)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 360, 640))
    assert torch.allclose(output["action_reread_delta_effective"], state.reread * output["action_reread_delta_raw"])
    assert torch.allclose(output["reason_reread_delta_effective"], state.reread * output["reason_reread_delta_raw"])
    assert torch.allclose(output["annotation_delta_effective"], state.annotation * output["annotation_delta_raw"])
    assert torch.count_nonzero(output["action_exchange_delta_effective"]) == 0
    assert torch.count_nonzero(output["reason_latent_delta_effective"]) == 0
    assert torch.allclose(output["threshold_logit_effective"], state.threshold * output["threshold_logit_raw"])
    assert torch.allclose(
        output["action_logits_deploy"],
        output["action_logits_final_raw"] - output["threshold_logit_effective"][:4],
    )
    assert output["curriculum_state"]["stage"] == "grounding_ramp_1"
