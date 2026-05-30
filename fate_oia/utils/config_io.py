from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {p}")
    return data


def collect_cli_overrides(argv: list[str] | None) -> set[str]:
    out: set[str] = set()
    if not argv:
        return out
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].replace("-", "_")
        if key:
            out.add(key)
    return out


_KEY_MAP = {
    "paths.data_root": "data_root",
    "paths.raw_root": "raw_root",
    "paths.pretrained_weights": "pretrained_weights",
    "paths.grounding_cache_jsonl": "grounding_cache_jsonl",
    "paths.reason_grounding_rules": "reason_grounding_rules",
    "model.action_dim": "action_dim",
    "model.reason_dim": "reason_dim",
    "model.image_height": "image_height",
    "model.image_width": "image_width",
    "model.patch_size": "patch_size",
    "model.n_last_blocks": "n_last_blocks",
    "model.use_label_query": "use_label_query",
    "model.evidence_pooler_version": "evidence_pooler_version",
    "model.max_evidence_units_per_image": "max_evidence_units_per_image",
    "model.per_reason_topk_evidence": "per_reason_topk_evidence",
    "model.allow_fallback_counterfactual": "allow_fallback_counterfactual",
    "model.fallback_quality_multiplier": "fallback_quality_multiplier",
    "model.action.action_residual_enabled": "action_residual_enabled",
    "model.action.action_update_scale_init": "action_update_scale_init",
    "model.action.action_update_scale_max": "action_update_scale_max",
    "training.resume_checkpoint": "resume_checkpoint",
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.gradient_accumulation_steps": "gradient_accumulation_steps",
    "training.num_workers": "num_workers",
    "training.device": "device",
    "training.log_every": "log_every",
    "training.grad_clip_norm": "grad_clip_norm",
    "training.best_selection_split": "best_selection_split",
    "training.best_selection_metric": "best_selection_metric",
    "training.no_metric_early_stop": "no_metric_early_stop",
    "optimizer.lr_main": "lr",
    "optimizer.weight_decay": "weight_decay",
    "scheduler.patience": "plateau_patience",
    "scheduler.factor": "plateau_factor",
    "scheduler.min_lr": "plateau_min_lr",
    "scheduler.max_restores": "plateau_max_restores",
    "scheduler.monitor": "plateau_monitor",
    "compression.token_compression": "token_compression",
    "compression.compression_start_epoch": "compression_start_epoch",
    "compression.compression_warmup_epochs": "compression_warmup_epochs",
    "compression.compression_keep_ratio_start": "compression_keep_ratio_start",
    "compression.compression_keep_ratio_final": "compression_keep_ratio_final",
    "compression.num_summary_tokens": "num_summary_tokens",
    "compression.min_tokens": "min_tokens",
    "compression.token_score_mode": "token_score_mode",
    "loss.asl_gamma_pos": "asl_gamma_pos",
    "loss.asl_gamma_neg": "asl_gamma_neg",
    "loss.asl_clip": "asl_clip",
    "loss.action_visual_aux": "loss_action_visual",
    "loss.reason_to_action_gt": "loss_r2a_gt",
    "loss.action_preserve": "loss_action_preserve",
    "loss.action_update_penalty": "loss_action_update_penalty",
    "loss.evidence_quality": "loss_evidence_quality",
    "loss.evidence_sparsity": "loss_evidence_sparsity",
    "loss.direct_effect": "loss_direct_effect",
    "loss.context_suppression": "loss_context_suppression",
    "loss.evidence_sufficiency": "loss_evidence_sufficiency",
    "loss.non_target_preserve": "loss_non_target_preserve",
    "loss.replacement_negative": "loss_replacement",
    "loss.replacement_contrast": "loss_replacement_contrast",
    "loss.tail_logit_rank": "loss_tail_logit_rank",
    "loss.tail_direct_effect_rank": "loss_tail_causal_rank",
    "loss.sigmoid_f1_reason": "loss_sigmoid_f1",
    "loss.gate_l1": "loss_gate_l1",
    "counterfactual.enabled": "counterfactual_enabled",
    "counterfactual.start_epoch": "counterfactual_start_epoch",
    "counterfactual.ramp_epochs": "counterfactual_ramp_epochs",
    "counterfactual.cf_max_positive_reasons_per_sample": "cf_max_positive_reasons_per_sample",
    "counterfactual.cf_max_views_per_batch": "cf_max_views_per_batch",
    "counterfactual.common_margin": "cf_common_margin",
    "counterfactual.tail_margin": "cf_tail_margin",
    "counterfactual.action_margin": "cf_action_margin",
    "counterfactual.context_margin": "cf_context_margin",
    "counterfactual.sufficiency_margin": "cf_sufficiency_margin",
    "replacement.enabled": "replacement_enabled",
    "replacement.memory_bank_size": "memory_bank_size",
    "tail.labels": "tail_labels",
    "calibration.enabled": "calibration_enabled",
}


def _walk(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for k, v in value.items():
            rows.extend(_walk(f"{prefix}.{k}" if prefix else str(k), v))
        return rows
    return [(prefix, value)]


def flatten_config_for_args(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, value in _walk("", config):
        key = _KEY_MAP.get(path)
        if key:
            out[key] = value
        elif "." not in path and path not in {"config_version", "experiment_name"}:
            out[path] = value
    if "config_version" in config:
        out["config_version"] = config["config_version"]
    if "experiment_name" in config:
        out["experiment_name"] = config["experiment_name"]
    return out


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    required_config_version: str | None = None,
) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    initial, _ = parser.parse_known_args(argv_list)
    overrides = collect_cli_overrides(argv_list)
    config_path = getattr(initial, "config", None)
    config_data: dict[str, Any] = {}
    if config_path:
        config_data = load_yaml_config(config_path)
        flat = flatten_config_for_args(config_data)
        defaults = {k: v for k, v in flat.items() if k not in overrides and hasattr(initial, k)}
        if defaults:
            parser.set_defaults(**defaults)
    args = parser.parse_args(argv_list)
    args.config_data = config_data
    args.config_source = str(config_path) if config_path else ""
    args.cli_overrides = sorted(overrides)
    args.config_version = config_data.get("config_version", getattr(args, "config_version", ""))
    args.experiment_name = config_data.get("experiment_name", getattr(args, "experiment_name", ""))
    if required_config_version and args.config_version != required_config_version:
        raise ValueError(f"Expected config_version={required_config_version}, got {args.config_version!r}")
    return args


def write_resolved_config(args: argparse.Namespace, output_dir: str | Path) -> dict[str, Any]:
    import yaml

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in vars(args).items() if _jsonable(v)}
    (out / "config_resolved.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return data


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
