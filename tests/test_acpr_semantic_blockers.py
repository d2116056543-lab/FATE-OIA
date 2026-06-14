import torch
import yaml

from fate_oia.losses.acpr_losses import matched_pair_logit_loss, partial_label_reason_loss, action_combo_drop_add_loss
from fate_oia.models.acpr_pair_memory import ACPRPairMemory
from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def test_acpr_reason_grammar_matches_external_names():
    grammar = yaml.safe_load(open("configs/acpr_reason_predicate_grammar.yaml", encoding="utf-8"))
    names = yaml.safe_load(open("configs/bdd_oia_reason_names_external.yaml", encoding="utf-8"))["names"]
    assert all(grammar["reasons"][i]["name"] == names[i] for i in range(21))


def test_acpr_predicate_target_builder_geometry_sources():
    b = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    out = b.build_from_records([{"labels": [
        {"category": "car", "box2d": {"x1": 600, "y1": 400, "x2": 720, "y2": 680}},
        {"category": "lane", "poly2d": [[{"x": 230, "y": 500}, {"x": 200, "y": 700}]]},
    ], "drivable_available": True}])
    assert out["predicate_targets"].sum() > 0
    assert out["predicate_coverage"]["object_box_count"] > 0
    assert out["predicate_coverage"]["lane_poly_count"] > 0
    assert out["predicate_coverage"]["drivable_count"] > 0


def test_acpr_pair_mining_reason_specific_contract():
    m = ACPRPairMemory()
    emb = torch.randn(4, 384)
    action = torch.tensor([[1,0,0,0],[1,0,0,0],[0,1,0,0],[1,0,0,0]]).float()
    reason = torch.zeros(4, 21); reason[0, 5] = 1; reason[2, 5] = 1
    pred = torch.rand(4, 32)
    contradiction = torch.zeros(4, 21); contradiction[:, 5] = torch.tensor([0.0, 0.9, 0.1, 0.8])
    pairs = m.mine_pairs(emb, action, reason, [5], global_embedding=emb, predicate_probs=pred, contradiction_scores=contradiction)
    assert {"pair_pos_indices", "pair_neg_indices", "pair_reason_ids", "pair_weights", "pair_contradiction"} <= set(pairs)
    assert pairs["pair_reason_ids"].numel() > 0


def test_acpr_pair_loss_only_updates_selected_reason():
    logits = torch.zeros(3, 21, requires_grad=True)
    pairs = {"pair_pos_indices": torch.tensor([0]), "pair_neg_indices": torch.tensor([1]), "pair_reason_ids": torch.tensor([5]), "pair_weights": torch.tensor([1.0])}
    loss = matched_pair_logit_loss(logits, pairs)
    loss.backward()
    assert logits.grad[:, 5].abs().sum() > 0
    assert logits.grad[:, [i for i in range(21) if i != 5]].abs().sum() == 0


def test_acpr_pu_reason_loss_uses_contradiction_weights():
    logits = torch.zeros(2, 2)
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    low = partial_label_reason_loss(logits, target, torch.zeros_like(target))
    high = partial_label_reason_loss(logits, target, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert high > low


def test_acpr_action_combo_drop_add_explicit_stats():
    logits = torch.zeros(2, 16)
    action = torch.tensor([[1,0,0,1], [1,0,1,0]]).float()
    loss, stats = action_combo_drop_add_loss(logits, action, return_stats=True)
    assert loss >= 0
    assert "drop_margin_mean" in stats and "add_margin_mean" in stats
