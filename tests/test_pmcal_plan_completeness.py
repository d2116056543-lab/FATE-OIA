from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_pmcal_trainer_has_real_train_calib_teacher_and_artifacts():
    src = (ROOT / "fate_oia/engine/train_pmcal_v2_oia.py").read_text(encoding="utf-8")
    assert "make_train_calib_indices" in src
    assert "train_calib_loader" in src
    assert "collect_threshold_teacher_pmcal" in src
    assert "checkpoint_best_test_base_fixed.pth" in src
    assert "best_epoch_source.json" in src
    assert "failure_cases.jsonl" in src
    assert '{"epoch": epoch, "available": True}' not in src
    assert "structured_records=None" not in src


def test_pmcal_geometry_builder_uses_structured_bdd100k_records_and_masks_test():
    from fate_oia.models.pmcal_predicate_observation_builder import PMCalPredicateObservationBuilder

    builder = PMCalPredicateObservationBuilder(scene_config=ROOT / "configs/acpr_scene_predicates.yaml")
    rec = {
        "labels": [
            {"category": "car", "box2d": {"x1": 270, "y1": 210, "x2": 380, "y2": 350}},
            {"category": "pedestrian", "box2d": {"x1": 300, "y1": 160, "x2": 330, "y2": 260}},
        ],
        "lane": [{"poly2d": [{"vertices": [[250, 260], [270, 350]]}]}],
        "drivable": [{"poly2d": [{"vertices": [[200, 260], [440, 260], [500, 360], [140, 360]]}]}],
    }
    train = builder.build(
        batch_size=1,
        device=torch.device("cpu"),
        split="train",
        reason_labels=torch.zeros(1, 21),
        structured_records=[rec],
    )
    assert float(train["obs_geometry_mask"].sum()) > 0.0
    assert float(train["obs_geometry_value"].sum()) > 0.0
    assert train["source_stats"]["geometry_positive_count"] > 0
    test = builder.build(
        batch_size=1,
        device=torch.device("cpu"),
        split="test",
        reason_labels=torch.ones(1, 21),
        structured_records=[rec],
    )
    assert float(test["obs_geometry_mask"].sum()) == 0.0
    assert float(test["obs_reason_mask"].sum()) == 0.0


def test_pmcal_conflict_optimizer_projects_conflicting_groups():
    from fate_oia.optim.pmcal_conflict_aware_optimizer import PMCalConflictAwareOptimizer

    w = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([w], lr=0.1)
    wrapper = PMCalConflictAwareOptimizer(opt, shared_params=[w])
    l1 = w.sum()
    l2 = -w.sum()
    wrapper.step_losses({"a": l1, "b": l2})
    assert wrapper.last_stats["projection_applied_count"] >= 1
    assert torch.isfinite(w).all()


def test_pmcal_certified_pair_loss_is_near_boundary_and_reliability_gated():
    from fate_oia.losses.pmcal_certified_pair_loss import certified_near_boundary_pair_loss

    reason_logits = torch.tensor([[0.05, 4.0], [-0.04, -4.0], [3.0, 0.0]])
    labels = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    reliable = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    loss, stats = certified_near_boundary_pair_loss(
        reason_logits,
        labels,
        reliable_mask=reliable,
        margin=0.2,
        boundary=0.2,
    )
    assert stats["near_boundary_pair_count"] > 0
    assert stats["certified_pair_count"] > 0
    assert stats["reason_specific_pairs"] >= 1
    far_loss, far_stats = certified_near_boundary_pair_loss(
        reason_logits * 20,
        labels,
        reliable_mask=reliable,
        margin=0.2,
        boundary=0.2,
    )
    assert far_stats["near_boundary_pair_count"] == 0
    assert float(far_loss) == 0.0


def test_pmcal_audit_rejects_placeholder_artifacts_and_requires_plan_hard_gates():
    audit = (ROOT / "fate_oia/engine/audit_pmcal_v2_implementation.py").read_text(encoding="utf-8")
    assert "placeholder_artifact_checks" in audit
    assert "train_calib_teacher_checks" in audit
    assert "geometry_dynamic_checks" in audit
    assert "conflict_projection_dynamic_checks" in audit
    assert "checkpoint_artifact_checks" in audit
