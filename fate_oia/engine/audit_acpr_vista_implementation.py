from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch
import yaml

from fate_oia.models.acpr_oia_model import ACPROIAModel


REQUIRED_FILES = [
    "configs/fate_oia_train_360x640_acpr_vista_v1.yaml",
    "fate_oia/models/acpr_visual_token_adapter.py",
    "fate_oia/utils/acpr_vista_gradient_coordinator.py",
    "fate_oia/utils/acpr_pair_budget.py",
    "fate_oia/utils/acpr_teacher_lock.py",
    "fate_oia/utils/acpr_vista_training_control.py",
    "fate_oia/utils/acpr_vista_artifacts.py",
    "fate_oia/engine/audit_acpr_vista_implementation.py",
    "fate_oia/engine/audit_acpr_vista_gates.py",
    "fate_oia/engine/probe_acpr_vista_memory.py",
    "fate_oia/engine/eval_acpr_vista_faithfulness.py",
    "fate_oia/engine/export_acpr_vista_visuals.py",
    "fate_oia/engine/supervise_acpr_vista_foreground.py",
    "scripts/FATE_OIA_acpr_vista_v1_foreground.ps1",
]

FORBIDDEN = [
    "acpr_predicate_action_coupling",
    "acpr_semantic_evidence_coattention",
    "acpr_triadic_mediator",
    "predicate_conditioned_threshold",
    "predicate_filtered_hardpair",
    "acpr_action_candidates",
    "acpr_action_utility",
    "acpr_fusionlite",
    "FrozenRunC",
    "frozen_run_c",
    "cached_logits",
    "tail_residual_adapter",
    "MoE",
    "specialist",
    "graph_delta_to_logits: true",
    "action_set_affects_final_action: true",
    "feature_cache_enabled: true",
    "token_compression: keep_merge",
    "best_selection_split: val",
    "eval_splits: val",
    "checkpoint_best_val",
    "Start-Process",
    "Start-Job",
    "nohup",
    "daemon",
    "scheduled task",
    "hidden cmd",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def audit(config: str, output_dir: str, device: str = "cpu", write_review_pass: bool = False) -> dict:
    root = Path.cwd()
    missing = [p for p in REQUIRED_FILES if not (root / p).exists()]
    ast_errors = {}
    for p in REQUIRED_FILES:
        if p.endswith(".py") and (root / p).exists():
            try:
                ast.parse(read_text(root / p), filename=p)
            except SyntaxError as exc:
                ast_errors[p] = str(exc)
    scan_files = [
        p for p in REQUIRED_FILES
        if (root / p).exists()
        and p != "fate_oia/engine/audit_acpr_vista_implementation.py"
        and not p.endswith(".pth")
    ]
    scanned = "\n".join(read_text(root / p) for p in scan_files)
    forbidden_hits = {pat: (pat in scanned) for pat in FORBIDDEN}
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    model = ACPROIAModel(
        use_mock_dino=True,
        threshold_enabled=bool(cfg.get("threshold", {}).get("enabled", False)),
        vista_enabled=bool(cfg.get("vista", {}).get("enabled", False)),
        vista_kwargs={
            "rank": int(cfg.get("vista", {}).get("rank", 48)),
            "gate_floor": float(cfg.get("vista", {}).get("gate_floor", 0.20)),
            "detach_predicate_gate": bool(cfg.get("vista", {}).get("detach_predicate_gate", True)),
        },
    )
    x = torch.randn(2, 3, 360, 640)
    out = model(x, epoch=0)
    shape_checks = {
        "action_logits": list(out["action_logits_final_raw"].shape) == [2, 4],
        "reason_logits": list(out["reason_logits_final_raw"].shape) == [2, 21],
        "vista_enabled": bool(out.get("vista_enabled", False)),
        "vista_gate_map": list(out["vista_gate_map"].shape) == [2, 3600],
        "predicate_attention": list(out["predicate_attention"].shape)[-1] == 3600,
    }
    functional_checks = {
        "adapter_file": "ACPRPredicateAnchoredVisualAdapter" in read_text(root / "fate_oia/models/acpr_visual_token_adapter.py"),
        "local_geometry": "Conv2d" in read_text(root / "fate_oia/models/acpr_visual_token_adapter.py") and "groups=rank" in read_text(root / "fate_oia/models/acpr_visual_token_adapter.py"),
        "rezero": torch.allclose(model.visual_adapter.gate_raw.detach(), torch.zeros_like(model.visual_adapter.gate_raw.detach())),
        "model_forward": all(shape_checks.values()),
        "config_no_cache": cfg.get("experiment", {}).get("feature_cache_enabled") is False,
        "config_no_compression": cfg.get("experiment", {}).get("token_compression") == "none",
        "test_best": cfg.get("experiment", {}).get("best_selection_split") == "test",
        "dataloader_prefetch": all(
            token in read_text(root / "fate_oia/engine/train_acpr_oia.py")
            for token in ["persistent_workers", "prefetch_factor", "pin_memory"]
        ),
        "pair_memory_ring_buffer": all(
            token in read_text(root / "fate_oia/models/acpr_pair_memory.py")
            for token in ["_memory_cursor", "_memory_count", "index_copy_", "memory_device"]
        )
        and "torch.cat([old, payload[key]]" not in read_text(root / "fate_oia/models/acpr_pair_memory.py"),
        "pair_memory_cuda_config": cfg.get("pair_mining", {}).get("pair_memory_device") == "cuda",
        "vista_gates_not_placeholder": "placeholder" not in read_text(root / "fate_oia/engine/audit_acpr_vista_gates.py").lower(),
        "memory_probe_real_step": "real_cuda_forward_backward" in read_text(root / "fate_oia/engine/probe_acpr_vista_memory.py")
        and "schema probe" not in read_text(root / "fate_oia/engine/probe_acpr_vista_memory.py").lower(),
        "num_workers_parallel": int(cfg.get("data", {}).get("num_workers", 0)) >= 4,
    }
    passed = not missing and not ast_errors and not any(forbidden_hits.values()) and all(functional_checks.values())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pass": bool(passed),
        "checked_files": REQUIRED_FILES,
        "missing_items": missing,
        "ast_errors": ast_errors,
        "forbidden_patterns": forbidden_hits,
        "architecture_checks": functional_checks,
        "shape_checks": shape_checks,
        "warnings": [],
        "review_pass_path": str(out_dir / "REVIEW_PASS_ACPR_VISTA_V1.txt"),
    }
    (out_dir / "implementation_audit_ACPR_VISTA_V1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if passed and write_review_pass:
        (out_dir / "REVIEW_PASS_ACPR_VISTA_V1.txt").write_text("REVIEW_PASS_ACPR_VISTA_V1\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    payload = audit(args.config, args.output_dir, args.device, args.write_review_pass)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
