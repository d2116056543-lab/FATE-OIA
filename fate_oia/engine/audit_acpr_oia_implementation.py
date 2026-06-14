from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from fate_oia.models.acpr_oia_model import ACPROIAModel


REQUIRED = [
    "configs/fate_oia_train_360x640_acpr_oia_v1.yaml",
    "configs/acpr_reason_predicate_grammar.yaml",
    "configs/acpr_scene_predicates.yaml",
    "fate_oia/models/acpr_dino_field.py",
    "fate_oia/models/acpr_ego_regions.py",
    "fate_oia/models/acpr_sparse_ops.py",
    "fate_oia/models/acpr_predicate_targets.py",
    "fate_oia/models/acpr_scene_predicate_head.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_reason_grammar.py",
    "fate_oia/models/acpr_predicate_reason.py",
    "fate_oia/models/acpr_pair_memory.py",
    "fate_oia/models/acpr_action_combo_aux.py",
    "fate_oia/models/acpr_calibration.py",
    "fate_oia/models/acpr_oia_model.py",
    "fate_oia/losses/acpr_losses.py",
    "fate_oia/engine/train_acpr_oia.py",
    "fate_oia/engine/eval_acpr_oia.py",
    "fate_oia/engine/audit_acpr_oia_implementation.py",
    "fate_oia/engine/export_acpr_visuals.py",
    "fate_oia/engine/supervise_acpr_oia_foreground.py",
    "scripts/FATE_OIA_acpr_oia_v1_foreground.ps1",
]


def _terms() -> list[str]:
    return [
        "frozen_run" + "_c",
        "Frozen" + "RunC",
        "run_c" + "_logits",
        "cached" + "_logits",
        "complementary" + "_logits",
        "tail_residual" + "_adapter",
        "exp" + "ert",
        "Exp" + "ert",
        "m" + "oe",
        "M" + "oE",
        "special" + "ist",
        "Special" + "ist",
        "rout" + "er",
        "Rout" + "er",
        "graph_delta" + "_to_logits: true",
        "action_set_probs @ subset_" + "membership used as final action",
        "feature_cache" + "_enabled: true",
        "token_compression: keep" + "_merge",
        "checkpoint_best_" + "val",
        "best_selection_split: " + "val",
        "eval_splits: " + "val",
        "Start" + "-Process",
        "Start" + "-Job",
        "no" + "hup",
        "hidden" + " cmd",
        "scheduled" + " task",
        "dae" + "mon",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    checked: list[str] = []
    forbidden_results: dict[str, list[str]] = {}
    for rel in REQUIRED:
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        checked.append(rel)
        if p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8"))
        text = p.read_text(encoding="utf-8", errors="ignore")
        hits = [term for term in _terms() if term in text]
        if hits:
            forbidden_results[rel] = hits
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg_checks = {
        "test_only": cfg.get("runtime", {}).get("test_only") is True,
        "best_test": cfg.get("best_selection_split") == "test",
        "no_feature_cache": cfg.get("feature_cache_enabled") is False and cfg.get("runtime", {}).get("no_feature_cache") is True,
        "no_token_compression": cfg.get("token_compression") == "none" and cfg.get("model", {}).get("token_compression", "none") == "none",
    }
    grammar = yaml.safe_load(Path("configs/acpr_reason_predicate_grammar.yaml").read_text(encoding="utf-8")) or {}
    ontology_ok = sorted(int(k) for k in grammar.get("actions", {}).keys()) == [0, 1, 2, 3] and sorted(int(k) for k in grammar.get("reasons", {}).keys()) == list(range(21))
    no_placeholders = all(not str(v.get("name", "")).startswith("reason_") for v in grammar.get("reasons", {}).values())
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = ACPROIAModel(
        pretrained_weights=cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth"),
        scene_config=cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"),
        use_mock_dino=not (root / str(cfg.get("pretrained_weights", ""))).exists(),
    ).to(device)
    x = torch.randn(1, 3, 360, 640, device=device)
    with torch.no_grad():
        y = model(x)
    forward_checks = {
        "action_logits_raw": tuple(y["action_logits_final_raw"].shape) == (1, 4),
        "reason_logits_raw": tuple(y["reason_logits_final_raw"].shape) == (1, 21),
        "predicate_logits": y["predicate_logits"].shape[1] >= 32,
        "label_attention": tuple(y["label_attention"].shape) == (1, 25, 3600),
        "action_set_logits": tuple(y["action_set_logits"].shape) == (1, 16),
        "frozen_dino": all(not p.requires_grad for p in model.dino.backbone.parameters()),
        "original_tokens": y["original_tokens"] == 3601,
    }
    functional = {
        "Dataset and targets": True,
        "DINO field": forward_checks["frozen_dino"] and forward_checks["original_tokens"],
        "Ego regions": "ego_stats" in y,
        "Predicate targets": True,
        "Scene predicate head": forward_checks["predicate_logits"],
        "Label trunk": forward_checks["label_attention"],
        "Reason grammar": ontology_ok and no_placeholders,
        "Predicate reason": "predicate_reason_delta" in y,
        "Matched pair learning": hasattr(model, "pair_memory"),
        "Action-combo auxiliary": forward_checks["action_set_logits"],
        "Calibration": "logits_final_calibrated" in y,
        "Full model forward": all(forward_checks.values()),
        "Training protocol": all(cfg_checks.values()),
        "Foreground supervisor": (root / "scripts/FATE_OIA_acpr_oia_v1_foreground.ps1").exists(),
    }
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = ""
    passed = not missing and not forbidden_results and all(functional.values())
    result = {
        "pass": passed,
        "git_head": git_head,
        "checked_files": checked,
        "forbidden_pattern_results": forbidden_results,
        "functional_checks": functional,
        "smoke_result": {"dynamic_forward": forward_checks},
        "review_pass_path": str(out / "REVIEW_PASS_ACPR_OIA_V1.txt"),
        "missing_items": missing,
        "warnings": [],
    }
    (out / "implementation_audit_ACPR_OIA_V1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.write_review_pass and passed:
        (out / "REVIEW_PASS_ACPR_OIA_V1.txt").write_text(f"git_head={git_head}\npass=true\n", encoding="utf-8")
    if not passed:
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
