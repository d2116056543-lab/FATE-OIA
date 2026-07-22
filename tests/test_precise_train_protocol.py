from pathlib import Path

import torch
import yaml

from fate_oia.engine.train_precise_oia import _slice_batch_output, build_optimizers
from fate_oia.models.precise_oia_model import PRECISEOIAModel


ROOT = Path(__file__).resolve().parents[1]


def test_train_protocol_is_test_only_with_no_metric_early_stop():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["eval_splits"] == "test"
    assert config["best_selection_split"] == "test"
    assert config["training"]["no_metric_early_stop"] is True
    assert config["pu"] == {"enabled": False, "weight": 0.0}


def test_main_representation_loss_cannot_train_threshold_head():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "deploy_loss" not in source
    assert "for calib_batch in calib_loader" in source


def test_all_planned_mechanism_losses_are_called_by_trainer():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    for required_call in (
        "packed_target_specific_interventions(",
        "two_way_consistency_loss(",
        "refinement_loss(",
        "update_view_consistency(",
    ):
        assert required_call in source
    assert "export_precise_cases(" in source


def test_non_main_losses_share_update_based_warmup_and_required_diagnostics_are_logged():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "latent_regularizer = 0.15 * 0.02 * losses[\"loss_evidence_latent_diversity\"]" in source
    assert "main_loss = losses[\"loss_total\"] - 0.15 * losses[\"loss_evidence\"] - latent_regularizer" in source
    assert "total_loss = main_loss + auxiliary_warmup * (" in source
    assert "0.15 * losses[\"loss_evidence\"]" in source
    assert "+ latent_regularizer" in source
    for field in (
        "action_reread_to_direct_ratio",
        "reason_reread_to_direct_ratio",
        "hard_sample_improvement_rate",
        "easy_sample_regression_rate",
        "grad_evidence_grounding",
        "grad_evidence_target_credit_raw",
        "grad_evidence_target_credit_projected",
    ):
        assert field in source


def test_pilot_supervisor_calls_pcvl_and_full_gate_is_hash_bound():
    source = (ROOT / "fate_oia" / "engine" / "supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    assert "run_precise_pcvl" in source
    assert "verify_review_hash" in source


def test_resume_restores_model_optimizers_schedulers_and_rng():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "load_resume_checkpoint(" in source
    for state in ("cuda_rng_state", "python_rng_state", "global_optimizer_step", "global_micro_step", "expected_fingerprint"):
        assert state in source


def test_pilot_resume_restores_pcvl_probe_and_optimizer_state():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert '"pcvl_probes"' in source
    assert '"pcvl_optimizer"' in source
    assert "pcvl_probes.load_state_dict" in source
    assert "pcvl_optimizer.load_state_dict" in source


def test_sample_limited_smoke_uses_full_train_metadata_for_field_preflight():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "full_train_set = BDDOIAMultiTaskDataset" in source
    assert "build_train_grounding_targets(full_train_set" in source


def test_configured_bf16_clipping_and_owner_diagnostics_are_executed():
    source = (ROOT / "fate_oia" / "engine" / "train_precise_oia.py").read_text(encoding="utf-8")
    assert "torch.autocast(" in source
    assert "clip_grad_norm_(" in source
    for field in ("grad_norm_pre_clip", "grad_norm_post_clip", "parameter_delta_norm", "optimizer_step_count"):
        assert field in source


def test_norm_bias_and_embedding_parameters_have_zero_weight_decay():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    model = PRECISEOIAModel(use_mock_dino=True)
    optimizers = build_optimizers(model, config)
    assert set(optimizers) == {"action_foundation", "action_decoder", "reason_semantic", "evidence_core", "exchange_reread", "annotation_adapter", "threshold_head"}
    assert all(any(group["weight_decay"] == 0.0 for group in optimizer.param_groups) for optimizer in optimizers.values())


def test_batch_output_slicing_never_slices_schema_tensors_when_sizes_coincide():
    full_batch = 10
    output = {
        "action_logits_final_raw": torch.randn(full_batch, 4),
        "evidence_view_consistency": torch.linspace(0.1, 1.0, 10),
        "action_evidence_family_mask": torch.ones(4, 10, dtype=torch.bool),
        "evidence_part_valid": torch.ones(10, 8, dtype=torch.bool),
        "evidence_geometry_type": torch.arange(10),
        "branch_logits": {"reason_semantic": torch.randn(full_batch, 21)},
    }

    canonical = _slice_batch_output(output, 8, full_batch)
    mirrored = _slice_batch_output(output, full_batch, full_batch, start=8)

    assert canonical["action_logits_final_raw"].shape == (8, 4)
    assert mirrored["action_logits_final_raw"].shape == (2, 4)
    assert canonical["branch_logits"]["reason_semantic"].shape == (8, 21)
    assert mirrored["branch_logits"]["reason_semantic"].shape == (2, 21)
    for key in ("evidence_view_consistency", "action_evidence_family_mask", "evidence_part_valid", "evidence_geometry_type"):
        assert torch.equal(canonical[key], output[key])
        assert torch.equal(mirrored[key], output[key])


def test_runtime_profiles_are_memory_isolated_and_persisted_before_selection():
    source = (ROOT / "fate_oia" / "engine" / "profile_precise_oia.py").read_text(encoding="utf-8")
    assert "gc.collect()" in source
    assert source.count("torch.cuda.empty_cache()") >= 2
    assert source.index('(root / "runtime_profile.json").write_text') < source.index("selected = choose_runtime_profile")
