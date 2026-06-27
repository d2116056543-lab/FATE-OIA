from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel


FORBIDDEN_PATTERNS = [
    "ACPROIAModel(",
    "BDD-OIA",
    "bdd_oia_multitask",
    "train_acpr_oia",
    "feature_cache_enabled: true",
    "token_compression: keep_merge",
    "checkpoint_best_val",
    "best_selection_split: val",
    "eval_splits: val",
    "Start-Process",
    "Start-Job",
    "nohup",
    "hidden cmd",
]


REQUIRED_FILES = [
    "fate_oia/acpr_interactflow/types.py",
    "fate_oia/acpr_interactflow/psi_damo_dataset.py",
    "fate_oia/acpr_interactflow/visual_encoder.py",
    "fate_oia/acpr_interactflow/dynamic_predicate_field.py",
    "fate_oia/acpr_interactflow/interaction_flow.py",
    "fate_oia/acpr_interactflow/decision_ledger.py",
    "fate_oia/acpr_interactflow/exp29_head.py",
    "fate_oia/acpr_interactflow/model.py",
    "fate_oia/losses/acpr_interactflow_losses.py",
    "fate_oia/engine/train_acpr_interactflow_psi.py",
    "fate_oia/engine/eval_acpr_interactflow_psi.py",
    "configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml",
]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _scan_forbidden(root: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    scan_paths = list((root / "fate_oia" / "acpr_interactflow").rglob("*.py"))
    scan_paths += [
        root / "fate_oia" / "losses" / "acpr_interactflow_losses.py",
        root / "fate_oia" / "engine" / "train_acpr_interactflow_psi.py",
        root / "fate_oia" / "engine" / "eval_acpr_interactflow_psi.py",
        root / "fate_oia" / "engine" / "run_acpr_interactflow_preflight.py",
        root / "fate_oia" / "engine" / "profile_acpr_interactflow.py",
        root / "fate_oia" / "engine" / "supervise_acpr_interactflow_foreground.py",
        root / "fate_oia" / "engine" / "export_acpr_interactflow_visuals.py",
        root / "fate_oia" / "engine" / "build_acpr_interactflow_atlas.py",
        root / "fate_oia" / "explain" / "acpr_interactflow_renderer.py",
        root / "fate_oia" / "explain" / "acpr_interactflow_atlas.py",
        root / "fate_oia" / "explain" / "acpr_interactflow_faithfulness.py",
        root / "configs" / "acpr_interactflow_pp_v1_psi_damo_11902.yaml",
        root / "configs" / "acpr_interactflow_predicates.yaml",
        root / "configs" / "acpr_interactflow_text_rules.yaml",
        root / "configs" / "acpr_interactflow_state_grammar.yaml",
        root / "scripts" / "FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1",
    ]
    for path in scan_paths:
        if not path.exists():
            continue
        if path.name == "audit_acpr_interactflow.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [p for p in FORBIDDEN_PATTERNS if p in text]
        if hits:
            results[str(path.relative_to(root))] = hits
    return results


def run_audit(config: str, output_dir: str, device: str = "cpu", write_review_pass: bool = False) -> dict:
    root = Path.cwd()
    out = Path(output_dir)
    cfg = load_interactflow_config(config)
    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    forbidden = _scan_forbidden(root)
    functional: dict[str, bool | str] = {}
    functional["dataset_and_targets"] = "psi_damo_dataset" not in str(forbidden)
    functional["direct_image_no_cache"] = not cfg["data"].get("feature_cache_enabled", True)
    functional["test_only"] = cfg["evaluation"].get("eval_splits") == ["test"]
    functional["formal_target_frame_excluded"] = cfg["data"].get("formal_input_uses_target_frame") is False
    functional["exp29_unknown_policy"] = cfg["data"].get("all_zero_exp29_is_unknown") is True
    train_source = (root / "fate_oia" / "engine" / "train_acpr_interactflow_psi.py").read_text(encoding="utf-8", errors="ignore")
    model_source = (root / "fate_oia" / "acpr_interactflow" / "model.py").read_text(encoding="utf-8", errors="ignore")
    loss_source = (root / "fate_oia" / "losses" / "acpr_interactflow_losses.py").read_text(encoding="utf-8", errors="ignore")
    preflight_source = (root / "fate_oia" / "engine" / "run_acpr_interactflow_preflight.py").read_text(encoding="utf-8", errors="ignore")
    intervention_source = (root / "fate_oia" / "acpr_interactflow" / "interventions.py").read_text(encoding="utf-8", errors="ignore")
    flow_source = (root / "fate_oia" / "acpr_interactflow" / "interaction_flow.py").read_text(encoding="utf-8", errors="ignore")
    transfer_source = (root / "fate_oia" / "acpr_interactflow" / "predicate_transfer.py").read_text(encoding="utf-8", errors="ignore")
    eval_source = (root / "fate_oia" / "engine" / "eval_acpr_interactflow_psi.py").read_text(encoding="utf-8", errors="ignore")
    export_source = (root / "fate_oia" / "engine" / "export_acpr_interactflow_visuals.py").read_text(encoding="utf-8", errors="ignore")
    supervisor_source = (root / "fate_oia" / "engine" / "supervise_acpr_interactflow_foreground.py").read_text(encoding="utf-8", errors="ignore")
    ps1_source = (root / "scripts" / "FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1").read_text(encoding="utf-8", errors="ignore")
    atlas_source = (root / "fate_oia" / "explain" / "acpr_interactflow_atlas.py").read_text(encoding="utf-8", errors="ignore")
    renderer_source = (root / "fate_oia" / "explain" / "acpr_interactflow_renderer.py").read_text(encoding="utf-8", errors="ignore")
    functional["formal_train_real_dino_only"] = "use_mock_dino=True" not in train_source and "--use_mock_dino" not in train_source
    functional["motion_uses_full15_fast_tokens"] = "self.motion(visual.fast_motion_tokens)" in model_source
    functional["oia_32_checkpoint_transfer"] = (
        "load_oia_predicate_queries" in transfer_source
        and "predicate_head.predicate_queries" in transfer_source
        and "source_checkpoint_sha256" in transfer_source
        and "source_oia_queries" in transfer_source
        and "oia_name_order_verified" in transfer_source
    )
    functional["text_transfer_uses_frozen_transformer"] = (
        "build_transformer_text_embeddings" in transfer_source
        and "AutoTokenizer.from_pretrained" in transfer_source
        and "AutoModel.from_pretrained" in transfer_source
        and "ontology_bow_fallback" in transfer_source
    )
    functional["eval_saves_ledger_middle_outputs"] = "ledger_gated_state_contributions" in eval_source and "logits_action_global" in eval_source
    functional["visual_export_real_ledger"] = "decision_ledger.png" in export_source and "Missing eval tensors" in export_source
    functional["atlas_renderer_tensor_linked"] = (
        "summarize_intervention_audit" in atlas_source
        and "decision_waterfall.png" in renderer_source
        and "identity_check_max_abs" in renderer_source
        and "ACPR-InteractFlow atlas:" not in (root / "fate_oia" / "engine" / "build_acpr_interactflow_atlas.py").read_text(encoding="utf-8", errors="ignore")
    )
    functional["state_temporal_losses_not_zero_placeholders"] = (
        "interaction_state_semantic_loss(output.flow.state_logits" in loss_source
        and "temporal_consistency_loss(" in loss_source
        and 'terms["interaction_state_semantic"] = output.action_logits.new_zeros' not in loss_source
        and 'terms["temporal_consistency"] = output.action_logits.new_zeros' not in loss_source
    )
    functional["response_lag_affects_flow"] = "lag_context" in flow_source and "factor_tokens = factor_tokens + 0.1 * lag_context" in flow_source
    functional["intervention_real_forward_probe"] = (
        "evaluate_intervention_suite" in intervention_source
        and "full_model_from_frames" in intervention_source
        and "downstream_recompute_from_formal_hook" in intervention_source
        and '"G_temporal_lag_necessity": True' not in preflight_source
    )
    functional["bf16_warmup_atomic_checkpoints"] = (
        "torch.autocast" in train_source
        and "_build_warmup_cosine_scheduler" in train_source
        and "_atomic_torch_save" in train_source
        and "checkpoint_best_action.pth" in train_source
        and "checkpoint_best_exp.pth" in train_source
        and "checkpoint_best_test.pth" in train_source
    )
    functional["required_epoch_artifact_schema"] = all(
        token in train_source
        for token in [
            "action_metrics.json",
            "exp29_metrics.json",
            "joint_metrics.json",
            "gradient_norms.json",
            "predicate_stats.json",
            "nnpu_calibration.json",
            "interaction_state_stats.json",
            "response_lag_stats.json",
            "decision_ledger_stats.json",
            "lightweight_interaction_influence.json",
            "predictions_action.jsonl",
            "predictions_exp29.jsonl",
            "fixed_case_intermediate_outputs.jsonl",
        ]
    )
    functional["required_run_root_artifact_schema"] = all(
        token in train_source
        for token in [
            "run_manifest.json",
            "config_resolved.yaml",
            "git_provenance.json",
            "oia_transfer_report.json",
            "optimizer_groups.json",
            "checkpoint_latest.pth",
            "checkpoint_best_action.pth",
            "checkpoint_best_exp.pth",
            "checkpoint_best_joint.pth",
            "checkpoint_best_test.pth",
            "run_complete.json",
        ]
    )
    functional["review_pass_head_remote_binding"] = (
        "Stale REVIEW_PASS" in supervisor_source
        and "git ls-remote" in ps1_source
        and "Worktree is dirty" in ps1_source
        and "GitHub branch HEAD mismatch" in ps1_source
    )
    functional["preflight_profile_uses_yaml_measured_batches"] = (
        "profile_batches = int(args.profile_batches if args.profile_batches is not None else cfg.get(\"profile\", {}).get(\"measured_batches\", 100))" in preflight_source
    )
    try:
        dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        pred_cfg = cfg["model"].get("predicates", {})
        model = ACPRInteractFlowPPModel(
            pretrained_weights=cfg["paths"]["dino_weights"],
            predicate_config="configs/acpr_interactflow_predicates.yaml",
            grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
            exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
            oia_acpr_checkpoint=cfg["paths"].get("oia_acpr_checkpoint"),
            text_encoder_model=cfg["paths"].get("text_encoder_model"),
            require_oia_transfer_source=bool(pred_cfg.get("require_oia_transfer_source", False)),
            require_transformer_text=bool(pred_cfg.get("require_transformer_text", False)),
            action_dim=int(cfg["data"]["action_dim"]),
            dino_chunk_size=int(cfg["model"]["visual_encoder"].get("dino_chunk_size", 2)),
            use_mock_dino=True,
        ).to(dev)
        frames = torch.randn(2, 15, 3, int(cfg["data"]["image_height"]), int(cfg["data"]["image_width"]), device=dev)
        output = model(frames, epoch=0)
        functional["model_forward"] = output.action_logits.shape == (2, int(cfg["data"]["action_dim"])) and output.exp29_logits.shape == (2, 29)
        output_lag0 = model(frames, epoch=0, intervention="lag_disabled")
        functional["lag_disabled_changes_forward_path"] = bool((output.action_logits - output_lag0.action_logits).abs().mean().detach().cpu() > 1e-8)
        output_pred_off = model(frames, epoch=0, intervention="predicate_off")
        functional["predicate_off_changes_forward_path"] = bool((output.action_logits - output_pred_off.action_logits).abs().mean().detach().cpu() > 1e-8)
        functional["dino_anchor_chunking"] = output.visual.stats.get("dino_chunk_size") == int(cfg["model"]["visual_encoder"].get("dino_chunk_size", 2))
        functional["predicate_field"] = output.predicates.predicate_logits.shape == (2, 48)
        functional["predicate_trajectory"] = (
            output.predicates.predicate_logits_trajectory.shape == (2, 15, 48)
            and output.predicates.predicate_probs_trajectory.shape == (2, 15, 48)
            and output.predicates.predicate_token_trajectory.shape[:3] == (2, 15, 48)
        )
        functional["predicate_evidence_geometry"] = (
            output.predicates.predicate_evidence_maps.shape == (2, 15, 48, 45, 80)
            and output.predicates.predicate_centroids.shape == (2, 15, 48, 2)
            and output.predicates.predicate_corridor_mass.shape == (2, 15, 48, 4)
        )
        functional["predicate_transfer_gate_surface"] = output.predicates.transfer_gate.shape[0] == 48
        transfer_report = model.predicates.transfer.report()
        functional["predicate_transfer_source_report"] = (
            transfer_report.get("source_loaded") is True
            and transfer_report.get("source_tensor_key") == "predicate_head.predicate_queries"
            and transfer_report.get("source_shape") == [32, 384]
            and transfer_report.get("oia_name_order_verified") is True
            and len(transfer_report.get("loaded_predicate_names", [])) == 32
        )
        functional["predicate_transfer_text_report"] = transfer_report.get("text_embedding_source") == "transformers_frozen"
        functional["decision_identity"] = float(output.ledger.identity_error.detach().cpu()) < 1e-5
        functional["ledger_exact_contribution_chain"] = (
            output.ledger.global_logits.shape == output.action_logits.shape
            and output.ledger.gated_state_contributions.shape[-1] == int(cfg["data"]["action_dim"])
            and output.ledger.benefit_gate.shape == output.ledger.gate.shape
        )
        functional["state_bank_surface"] = "state_group_logits" in output.aux and "state_layer_weights" in output.aux
        smoke_result = {"forward_ok": True, "action_shape": list(output.action_logits.shape), "exp29_shape": list(output.exp29_logits.shape)}
    except Exception as exc:
        functional["model_forward"] = False
        smoke_result = {"forward_ok": False, "error": repr(exc)}
    gates_summary_path = out / "preflight_gates_summary.json"
    gates_summary = {}
    if gates_summary_path.exists():
        try:
            gates_summary = __import__("json").loads(gates_summary_path.read_text(encoding="utf-8")).get("gates", {})
        except Exception:
            gates_summary = {}
    gate_evidence = {
        "real_dino_smoke": (out / "real_dino_smoke.json").exists(),
        "throughput_memory_profile": (out / "throughput_memory_profile.json").exists(),
        "gate_summary": (out / "preflight_gates_summary.json").exists(),
        "all_A_to_K_gates_true": bool(gates_summary) and all(bool(v) for v in gates_summary.values()),
    }
    if write_review_pass:
        functional.update({f"evidence_{k}": v for k, v in gate_evidence.items()})
    passed = not missing and not forbidden and all(bool(x) for x in functional.values())
    review_path = out / "REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt"
    report = {
        "pass": passed,
        "git_head": _git_head(),
        "checked_files": REQUIRED_FILES,
        "missing_items": missing,
        "forbidden_pattern_results": forbidden,
        "functional_checks": functional,
        "smoke_result": smoke_result,
        "gate_evidence": gate_evidence,
        "review_pass_path": str(review_path),
        "warnings": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "implementation_audit_ACPR_INTERACTFLOW_PP_V1.json", report)
    if write_review_pass and passed:
        review_path.write_text(json.dumps({"git_head": report["git_head"], "pass": True}, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.config, args.output_dir, args.device, args.write_review_pass)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
