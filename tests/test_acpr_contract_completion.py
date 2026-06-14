import inspect
from pathlib import Path

import torch

from fate_oia.losses import acpr_losses as losses
from fate_oia.models.acpr_action_combo_aux import ACPRActionComboAux
from fate_oia.models.acpr_calibration import ACPRCalibrationHead
from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.acpr_pair_memory import ACPRPairMemory
from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar


def test_dino_field_exposes_last_layer_tokens():
    model = ACPRDinoFieldExtractor(use_mock_dino=True)
    out = model(torch.randn(2, 3, 360, 640))
    assert out["patch_tokens_by_layer"].shape == (2, 3, 3600, 384)
    assert out["cls_tokens_by_layer"].shape == (2, 3, 384)
    assert out["patch_tokens_last"].shape == (2, 3600, 384)
    assert out["cls_token_last"].shape == (2, 384)


def test_reason_grammar_exposes_required_matrices_and_tail_set():
    grammar = ACPRReasonGrammar("configs/acpr_reason_predicate_grammar.yaml")
    predicates = ["front_vehicle_close", "road_clear", "left_lane_boundary", "right_turn_region"]
    assert grammar.tail_indices == [12, 9, 5, 14, 6, 11, 10, 13]
    assert grammar.positive_matrix(predicates).shape == (21, len(predicates))
    assert grammar.contradiction_matrix(predicates).shape == (21, len(predicates))
    assert grammar.compatible_action_matrix().shape == (21, 4)
    assert grammar.hard_negative_matrix().shape == (21, 21)


def test_pair_memory_has_cross_batch_enqueue_and_mines_from_memory():
    memory = ACPRPairMemory(memory_size=16)
    emb = torch.randn(4, 384)
    action = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]).float()
    reason = torch.zeros(4, 21)
    reason[0, 5] = 1
    pred = torch.rand(4, 32)
    memory.enqueue(["a", "b", "c", "d"], emb, pred, action, reason)
    new_reason = torch.zeros(1, 21)
    new_reason[0, 5] = 1
    pairs = memory.mine(
        ["e"],
        emb[:1],
        pred[:1],
        action[:1],
        new_reason,
        torch.ones(1, 21),
        [5],
    )
    assert pairs["pair_count"] > 0
    assert pairs["pair_neg_memory_indices"].numel() == pairs["pair_pos_indices"].numel()


def test_action_combo_aux_outputs_cardinality_logits_and_stats():
    head = ACPRActionComboAux()
    out = head(torch.randn(2, 25, 384), torch.randn(2, 4))
    assert out["action_set_logits"].shape == (2, 16)
    assert out["cardinality_logits"].shape == (2, 5)
    assert "combo_stats" in out


def test_calibration_head_returns_split_outputs():
    head = ACPRCalibrationHead(action_dim=4, reason_dim=21)
    out = head(torch.randn(2, 4), torch.randn(2, 21))
    assert out["action_logits_calibrated"].shape == (2, 4)
    assert out["reason_logits_calibrated"].shape == (2, 21)
    assert out["bias_action"].shape == (4,)
    assert out["bias_reason"].shape == (21,)
    assert out["temperature_action"].shape == (4,)
    assert out["temperature_reason"].shape == (21,)


def test_model_forward_full_plan_contract_keys():
    model = ACPROIAModel(use_mock_dino=True)
    out = model(torch.randn(2, 3, 360, 640))
    required = {
        "patch_tokens_last",
        "cls_token_last",
        "contradiction_score",
        "required_support_score",
        "predicate_reason_stats",
        "reason_logits_calibrated",
        "action_logits_calibrated",
        "cardinality_logits",
    }
    assert required <= set(out)
    assert out["branch_logits"].keys() >= {"direct", "direct_plus_predicate", "raw", "calibrated"}
    assert torch.allclose(out["action_logits_raw"], out["action_logits_direct"])


def test_loss_function_signatures_match_plan():
    assert list(inspect.signature(losses.predicate_reason_alignment_loss).parameters)[:4] == [
        "predicate_probs",
        "reason_targets",
        "grammar_positive_matrix",
        "grammar_contradiction_matrix",
    ]
    assert list(inspect.signature(losses.matched_pair_logit_loss).parameters)[:5] == [
        "reason_logits",
        "pair_pos_idx",
        "pair_neg_idx",
        "pair_reason_idx",
        "pair_weights",
    ]
    assert list(inspect.signature(losses.cardinality_loss).parameters)[:2] == ["cardinality_logits", "action_target"]
    assert list(inspect.signature(losses.calibration_loss).parameters)[:4] == [
        "action_logits_cal",
        "reason_logits_cal",
        "action_targets",
        "reason_targets",
    ]
    assert inspect.signature(losses.action_combo_drop_add_loss).parameters["margin"].default == 0.25


def test_required_artifact_names_are_present_in_training_source():
    src = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
    for name in [
        "implementation_fingerprint.json",
        "logits_action_raw_test.pt",
        "logits_reason_raw_test.pt",
        "logits_action_calibrated_test.pt",
        "logits_reason_calibrated_test.pt",
        "predicate_logits_test.pt",
        "predicate_probs_test.pt",
        "pair_cases_test.jsonl",
        "pair_mining_stats.jsonl",
        "pair_margin_per_reason.json",
        "checkpoint_best_test_tail_mf1.pth",
    ]:
        assert name in src


def test_eval_and_visual_export_are_not_placeholders():
    eval_src = Path("fate_oia/engine/eval_acpr_oia.py").read_text(encoding="utf-8")
    visual_src = Path("fate_oia/engine/export_acpr_visuals.py").read_text(encoding="utf-8")
    for token in ["tail_reason", "predicate_group", "pair_margin", "action_composition"]:
        assert token in eval_src
    for token in ["matched_negative", "predicate_delta", "reason_margin", "report.html"]:
        assert token in visual_src
