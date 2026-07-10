from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from fate_oia.engine import train_acpr_mosaic_ad as trainer


def test_training_path_calls_every_core_mosaic_component() -> None:
    source = inspect.getsource(trainer.train_representation_epoch)
    required = {
        "_make_weak_views",
        "grounding_builder",
        "build_mosaic_factor_loss",
        "build_mosaic_state_loss",
        "selective.hide_observed_positives",
        "build_mosaic_reason_loss",
        "action_cross_image_ranking_loss",
        "posterior_weighted_reason_ranking_loss",
        "action_anchor.accumulate",
        "action_anchor.finalize",
        "action_queue.enqueue",
        "reason_queue.enqueue",
    }
    assert all(name in source for name in required)


def test_training_does_not_collapse_tasks_into_one_total_backward() -> None:
    source = inspect.getsource(trainer.train_representation_epoch)
    assert "total_loss.backward" not in source
    assert "action_anchor.accumulate" in source
    assert "action_anchor.finalize" in source
    assert "action_loss / grad_accum" in source
    assert "explanation_loss / grad_accum" in source


def test_calibration_pass_has_no_test_loader_or_test_labels() -> None:
    source = inspect.getsource(trainer.fit_calibrator)
    assert "test" not in source.lower()
    assert "model.eval()" in source
    assert "requires_grad_(False)" in source
    assert "train_calib" in source


def test_formal_model_builder_never_instantiates_old_acpr_path() -> None:
    source = inspect.getsource(trainer.build_model_components)
    assert "MOSAICADModel" in source
    assert "ACPROIAModel" not in source
    assert "ACPRPairMemory" not in source


def test_model_builder_wires_all_formal_mosaic_config_values() -> None:
    config_path = Path("configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml")
    config = trainer.load_config(config_path)
    config["model"]["use_mock_dino"] = True
    model, *_ = trainer.build_model_components(config, config_path, torch.device("cpu"))

    predicates = model.observable_predicates
    typed = predicates.typed_attention
    assert model.dino.selected_layers == tuple(config["backbone"]["selected_layers"])
    assert model.dino.patch_size == config["backbone"]["patch_size"]
    assert predicates.anchors_per_factor == config["model"]["anchors_per_factor"]
    assert typed.heads == config["model"]["typed_attention_heads"]
    assert typed.point_samples == config["model"]["point_samples"]
    assert typed.curve_samples == config["model"]["curve_samples"]
    assert typed.region_samples == config["model"]["region_samples"]
    assert predicates.prototype_bank.prior_scale_max == config["model"]["spatial_prior_scale_max"]
    assert predicates.prototype_bank.prior_dropout == config["model"]["spatial_prior_dropout"]
    assert model.state_composer.state_residual_cap == config["model"]["state_residual_cap"]


def test_calibrator_has_a_fixed_step_budget() -> None:
    signature = inspect.signature(trainer.fit_calibrator)
    assert "max_steps" in signature.parameters
    source = inspect.getsource(trainer.fit_calibrator)
    assert "for step in range(max_steps)" in source
    assert "fixed_order" in source
    assert "calibration_objective" in source
    config = trainer.load_config("configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml")
    assert config["calibration"]["steps_per_epoch"] == 100
    assert config["calibration"]["batch_size"] == 256
    assert config["calibration"]["surrogate_temperature"] == 0.20


def test_limited_train_and_calib_subsets_are_multilabel_stratified_without_image_decode() -> None:
    class Sample:
        def __init__(self, index: int) -> None:
            self.file_name = f"sample_{index:03d}.jpg"
            self.action = [0, 0, 0, 0]
            self.reason = [0] * 21

    class Dataset:
        def __init__(self) -> None:
            self.samples = [Sample(index) for index in range(80)]

    dataset = Dataset()
    for label in range(25):
        for offset in range(3):
            sample = dataset.samples[(label * 3 + offset) % len(dataset.samples)]
            if label < 4:
                sample.action[label] = 1
            else:
                sample.reason[label - 4] = 1
    selected = trainer._stratified_subset_indices(dataset, list(range(80)), 32, seed=20260710)
    assert len(selected) == 32
    counts = trainer._positive_counts(dataset, selected)
    assert all(count > 0 for count in counts)
    assert selected == trainer._stratified_subset_indices(dataset, list(range(80)), 32, seed=20260710)


def test_formal_config_rejects_scientific_contract_drift(tmp_path) -> None:
    config_path = Path("configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml")
    config = trainer.load_config(config_path)
    config["loss"]["reason"]["posterior_bce_weight"] = 0.99
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="posterior_bce_weight"):
        trainer.load_config(drifted)


def _stable_evaluation() -> dict:
    action = {
        "Act_mF1": 0.70,
        "Act_per_label_ap": [0.70, 0.70, 0.70, 0.70],
    }
    return {
        "metrics_summary": {
            "raw": {"Act_mF1": 0.70, "Exp_mAP": 0.40},
            "deploy_fixed": {"Exp_mF1": 0.40},
            "test_oracle_diagnostic": {"Act_mF1": 0.72},
        },
        "action_branch_metrics": {
            "raw": action,
            "visual": {"Act_mF1": 0.70},
        },
    }


def test_scientific_stop_rules_cover_missingness_and_prior_failures() -> None:
    history = []
    first = {
        "propensity_bound_rate": 0.95,
        "posterior_all_on_rate": 0.95,
        "posterior_all_off_rate": 0.0,
        "posterior_recovery_available": False,
        "posterior_recovery_improvement": 0.0,
        "factor_audit_available": True,
        "prior_to_full_ratio": 0.95,
    }
    assert trainer._epoch_stop_reasons(5, _stable_evaluation(), history, first) == []
    second = {**first, "posterior_recovery_available": True}
    reasons = trainer._epoch_stop_reasons(8, _stable_evaluation(), history, second)
    assert "propensity_at_bounds_gt_0p90_two_epochs" in reasons
    assert "latent_reason_posterior_all_on_or_all_off_two_epochs" in reasons
    assert "factor_prior_only_ge_0p90_full_two_audits" in reasons
    assert "posterior_recovery_not_better_than_zero_as_negative" in reasons


def test_launcher_is_foreground_and_binds_review_and_github_heads() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "FATE_OIA_acpr_mosaic_ad_v1_foreground.ps1"
    ).read_text(encoding="utf-8")
    for forbidden in ("Start-Process", "Start-Job", "Register-ScheduledTask", "-WindowStyle Hidden"):
        assert forbidden not in script
    assert "& $Python -u @Arguments" in script
    assert "acpr_mosaic_ad_v1_REVIEW_PASS.json" in script
    assert "github_sync_pass.json" in script
    assert "fresh_clone_head" in script
    assert "review.config_hash" in script
    assert "review.runtime_selection_hash" in script
    assert "Get-FileHash" in script
    assert "acpr_mosaic_ad_v1_artifact_smoke" in script
    assert '"--artifact_smoke_dir", $artifactSmokeDir' in script
    assert "Assert-NewRunDirectory" in script
