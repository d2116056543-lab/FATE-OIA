import torch

from fate_oia.engine.train_pact_oia_probe import compute_losses, counterfactual_due, load_config
from fate_oia.models.pact_oia_model import PACTOIAModel
from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue


def test_formal_loss_registry_and_backward_are_finite():
    model = PACTOIAModel(use_mock_dino=True)
    image = torch.randn(2, 3, 360, 640)
    output = model(image, semantic_share_license=0.5, action_scale=1.0, reason_budget=0.25)
    predicates, patches = output["semantic_predicate_logits"].shape[1], output["semantic_predicate_attention"].shape[-1]
    batch = {
        "action": torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]]),
        "reason": torch.stack((torch.arange(21) % 2, (torch.arange(21) + 1) % 2)).float(),
    }
    structured = {
        "predicate_target": torch.zeros(2, predicates),
        "predicate_target_mask": torch.ones(2, predicates),
        "predicate_counter_mask": torch.zeros(2, predicates),
        "predicate_reliability": torch.ones(2, predicates),
        "predicate_map_target": torch.full((2, predicates, patches), 1.0 / patches),
        "predicate_map_mask": torch.ones(2, predicates),
    }
    queue = PACTBalancedPairQueue(21)
    queue.enqueue(output["reason_logits_final_train"], batch["reason"], 0, output["contradiction_score"])
    queue.enqueue(-output["reason_logits_final_train"], 1 - batch["reason"], 0, output["contradiction_score"])
    cfg = load_config("configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml")
    total, rows, stats = compute_losses(output, batch, structured, cfg, queue, 0)
    assert torch.isfinite(total)
    assert {row["name"] for row in rows} == set(cfg["loss_weights"])
    total.backward()
    assert stats["labels_with_pairs"] == 21
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_counterfactual_runs_once_per_scheduled_optimizer_update():
    decisions = [counterfactual_due("pact", 3, micro, 5, 4) for micro in range(5)]
    assert decisions == [False, False, False, False, True]
    assert not counterfactual_due("control", 3, 4, 5, 4)
