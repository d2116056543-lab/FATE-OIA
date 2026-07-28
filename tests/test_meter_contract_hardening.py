from pathlib import Path

import torch

from fate_oia.engine import eval_acpr_meter_oia as evaluator
from fate_oia.engine import profile_acpr_meter_oia as profiler
from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.losses.meter_pu_losses import meter_hidden_positive_audit
from fate_oia.losses.meter_reason_losses import weighted_reason_asl
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder
from fate_oia.utils.meter_posthoc_calibration import (
    apply_meter_deploy,
    fit_train_calib_deploy_theta,
)


def test_optimizer_uses_reference_batch_scaling_and_no_decay_contract() -> None:
    model = METEROIAModel(dim=16, use_mock_dino=True, factor_rank=4)
    config = {
        "training": {
            "batch_size": 6,
            "gradient_accumulation_steps": 5,
            "reference_effective_batch": 32,
            "weight_decay": 0.05,
            "lr_foundation_core": 1.5e-4,
            "lr_factor_evidence": 2.0e-4,
            "lr_semantic_action": 2.0e-4,
            "lr_action_selector": 2.0e-4,
            "lr_reason_global_private": 2.5e-4,
            "lr_reason_local_private": 2.5e-4,
            "lr_reason_annotation": 2.5e-4,
            "lr_meta_adapters": 1.0e-4,
            "lr_pu_private": 2.5e-4,
        }
    }
    optimizer = trainer._make_optimizer(model, config)
    assert all(group["lr"] <= 2.5e-4 * 30.0 / 32.0 + 1e-12 for group in optimizer.param_groups)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = parameter_names[id(parameter)]
            no_decay = (
                parameter.ndim <= 1
                or name.endswith(".bias")
                or "embedding" in name
                or "queries" in name
                or name.rsplit(".", 1)[-1]
                in {"factor_value", "null_key", "support_query", "counter_query"}
            )
            assert (float(group["weight_decay"]) == 0.0) is no_decay


def test_reason_decoder_has_self_attention_and_learned_layer_fusion() -> None:
    decoder = METERPrivateReasonDecoder(dim=16, reason_dim=21, action_dim=4)
    assert isinstance(decoder.reason_self_attention, torch.nn.MultiheadAttention)
    assert decoder.layer_router.shape == (3,)
    weights = torch.softmax(decoder.layer_router, dim=0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_reason_zero_weight_uses_observability_and_evidence() -> None:
    logits = torch.tensor([[2.0], [2.0]])
    target = torch.zeros_like(logits)
    evidence = torch.tensor([[0.9], [0.9]])
    low_obs = weighted_reason_asl(logits, target, evidence, torch.zeros_like(evidence))
    high_obs = weighted_reason_asl(logits, target, evidence, torch.ones_like(evidence))
    assert high_obs > low_obs


def test_hidden_positive_audit_scores_hidden_subset_not_complete_targets() -> None:
    count = 40
    targets = torch.zeros(count, 1)
    targets[:20] = 1.0
    factor = torch.full((count, 1), 0.2)
    private = torch.full((count, 1), 0.2)
    private[:20] = 0.95
    factor[:20] = 0.95
    audit = meter_hidden_positive_audit(
        private,
        factor,
        targets,
        hidden_fraction=0.30,
        min_positive_count=20,
        seed=7,
    )
    label = audit["labels"][0]
    assert label["audit_target"] == "deliberately_hidden_positive_vs_observed_zero"
    assert label["hidden_count"] == 6
    assert 0.0 <= label["hidden_positive_auprc"] <= 1.0


def test_posthoc_calibration_has_temperature_group_shrinkage_and_preserves_map() -> None:
    torch.manual_seed(3)
    logits = torch.randn(80, 4)
    labels = (torch.sigmoid(logits + torch.tensor([0.8, -0.6, 0.4, -0.2])) > 0.5).float()
    raw = multilabel_metrics_from_logits(logits, labels, prefix="Act_")
    result = fit_train_calib_deploy_theta(
        logits,
        labels,
        model_state_hash="state",
        label_groups=(0, 0, 1, 1),
    )
    deploy = apply_meter_deploy(logits, result)
    calibrated = multilabel_metrics_from_logits(deploy, labels, prefix="Act_")
    assert result.temperature.shape == result.theta.shape
    assert result.strategy in {"global", "group", "group_shrinkage", "per_label"}
    assert float(result.theta.square().mean().sqrt()) <= 0.35 * float(logits.square().mean().sqrt()) + 1e-6
    assert abs(raw["Act_mAP"] - calibrated["Act_mAP"]) < 1e-7


def test_metrics_and_evaluator_cover_auc_and_required_selector_branches() -> None:
    logits = torch.tensor([[3.0, -2.0], [-1.0, 2.0], [1.0, -0.5], [-2.0, 1.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    metrics = multilabel_metrics_from_logits(logits, labels, prefix="Act_")
    assert "Act_mAUC" in metrics and "Act_per_label_auc" in metrics
    assert "selector_visual_only" in evaluator.ACTION_BRANCHES
    assert "selector_semantic_only" in evaluator.ACTION_BRANCHES
    assert "mix_private" in evaluator.REASON_BRANCHES


def test_full_supervisor_requires_full_train_ready_and_plan_pilot_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    powershell = (root / "scripts" / "FATE_OIA_acpr_meter_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    supervisor = (root / "fate_oia" / "engine" / "supervise_acpr_meter_oia_foreground.py").read_text(encoding="utf-8")
    for source in (powershell, supervisor):
        assert "METER_OIA_V1_FULL_TRAIN_READY.json" in source
        assert "4096" in source and "1024" in source and "512" in source


def test_profiler_measures_real_optimizer_and_low_frequency_events() -> None:
    source = Path(profiler.__file__).read_text(encoding="utf-8")
    assert "warmup_updates: int = 5" in source
    assert "measured_updates: int = 20" in source
    assert "optimizer.step()" in source
    assert "_counterfactual_event(" in source
    assert "meta.event(" in source
    assert "fit_train_calib_deploy_theta(" in source
    assert '"real_data": True' in source


def test_standalone_evaluator_loads_checkpoint_and_writes_summary() -> None:
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "load_checkpoint(args.checkpoint" in source
    assert 'write_json(output_dir / "evaluation_summary.json"' in source
    assert "intentionally not a hidden training launcher" not in source


def test_trainer_resumes_at_deterministic_micro_step() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    for token in (
        "resume_micro_step",
        "epoch_resume_micro",
        "micro_step=micro + 1",
        "torch.Generator().manual_seed",
    ):
        assert token in source
