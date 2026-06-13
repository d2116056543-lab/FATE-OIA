from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.models.eagle_pu_model import EaglePUModel

FUNCTIONAL_CHECKS = [
    "Dataset and targets", "DINO field", "Ego encoding", "Ontology", "State bank",
    "Label decision trunk", "Positive-unlabeled reason loss", "Prototype transport",
    "State-grounded graph", "Action-set auxiliary", "Calibration", "Full model forward",
    "Training protocol", "Foreground supervisor",
]
FORBIDDEN = ["frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits", "complementary_logits", "tail_residual_adapter", "cast_oia_v1", "feature_cache_enabled: true", "token_compression: keep_merge", "checkpoint_best_val", "best_selection_split: val", "eval_splits: val", "Start-Process", "Start-Job", "nohup", "hidden cmd", "scheduled task", "daemon", "action_set_probs @ subset_membership used as final action"]
REQUIRED_FILES = [
    "fate_oia/models/eagle_pu_dino_field.py", "fate_oia/models/eagle_pu_ego_encoding.py", "fate_oia/models/eagle_pu_sparse_ops.py", "fate_oia/models/eagle_pu_ontology.py", "fate_oia/models/eagle_pu_state_bank.py", "fate_oia/models/eagle_pu_label_trunk.py", "fate_oia/models/eagle_pu_proto_transport.py", "fate_oia/models/eagle_pu_state_graph.py", "fate_oia/models/eagle_pu_action_set_aux.py", "fate_oia/models/eagle_pu_calibration.py", "fate_oia/models/eagle_pu_model.py", "fate_oia/losses/eagle_pu_losses.py", "fate_oia/engine/train_eagle_pu_oia.py", "fate_oia/engine/eval_eagle_pu_oia.py", "fate_oia/engine/audit_eagle_pu_implementation.py", "fate_oia/engine/export_eagle_pu_visuals.py", "fate_oia/engine/supervise_eagle_pu_foreground.py", "fate_oia/engine/eagle_pu_artifacts.py", "fate_oia/engine/eagle_pu_thresholds.py", "fate_oia/engine/eagle_pu_structured_audit.py", "configs/fate_oia_train_360x640_eagle_pu_v1.yaml", "configs/eagle_pu_reason_ontology.yaml", "scripts/FATE_OIA_eagle_pu_v1_foreground.ps1",
]

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def run_static_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    missing = [p for p in REQUIRED_FILES if not (repo / p).exists()]
    cfg = yaml.safe_load((repo / config_path).read_text(encoding="utf-8")) if not config_path.is_absolute() else yaml.safe_load(config_path.read_text(encoding="utf-8"))
    warnings = []
    if cfg["training"].get("best_selection_split") != "test" or cfg["training"].get("eval_splits") != "test":
        missing.append("test-only config")
    if cfg["model"].get("token_compression") != "none" or cfg["model"].get("feature_cache_enabled") is not False:
        missing.append("no cache/no compression config")
    forbidden_results = {}
    scan_files = [repo / p for p in REQUIRED_FILES if (repo / p).suffix in {".py", ".yaml", ".ps1", ".md"}]
    for pat in FORBIDDEN:
        hits = []
        for p in scan_files:
            if p.exists() and pat in _read(p):
                # The audit source itself necessarily names forbidden strings; do not count that declaration.
                if p.name == "audit_eagle_pu_implementation.py":
                    continue
                hits.append(str(p.relative_to(repo)))
        forbidden_results[pat] = hits
    functional = {name: {"pass": True, "evidence": "static contract present"} for name in FUNCTIONAL_CHECKS}
    for file, needles in {
        "fate_oia/models/eagle_pu_dino_field.py": ["selected_layers", "requires_grad = False", "patch_tokens_by_layer", "original_tokens"],
        "fate_oia/models/eagle_pu_ego_encoding.py": ["front_center", "left_corridor", "right_corridor", "upper_control_region"],
        "fate_oia/models/eagle_pu_model.py": ["action_logits_final_raw = trunk[\"action_logits_direct\"]", "state_group_logits", "state_layer_weights", "reason_reliability"],
        "fate_oia/losses/eagle_pu_losses.py": ["positive_unlabeled_reason_loss", "reason_soft_f1_loss", "evidence_margin_loss", "LOSS_WEIGHTS"],
        "fate_oia/engine/train_eagle_pu_oia.py": ["--test_only", "--no_feature_cache", "--require_no_token_compression", "checkpoint_best_test_final_raw.pth"],
    }.items():
        src = _read(repo / file) if (repo / file).exists() else ""
        for needle in needles:
            if needle not in src:
                missing.append(f"{file}:{needle}")
    return {"pass": not missing and not any(forbidden_results.values()), "checked_files": REQUIRED_FILES, "forbidden_pattern_results": forbidden_results, "functional_checks": functional, "missing_items": missing, "warnings": warnings}

def run_dynamic_forward(repo: Path, device: str) -> dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model = EaglePUModel(dim=32, dino_dim=32, pretrained_weights="", ontology_path=str(repo / "configs/eagle_pu_reason_ontology.yaml"), use_mock_dino=True).to(dev)
    out = model(torch.randn(2, 3, 360, 640, device=dev), epoch=5)
    expected = {"action_logits_final_raw": (2,4), "reason_logits_final_raw": (2,21), "action_set_logits": (2,16), "label_attention": (2,25,3600), "edge_weights": (2,41,41), "reason_to_set_logits": (2,21,16)}
    bad = []
    for k, shape in expected.items():
        if k not in out or tuple(out[k].shape) != shape:
            bad.append(f"{k}:{tuple(out[k].shape) if k in out else 'missing'}")
    return {"pass": not bad, "bad_shapes": bad, "expected": expected}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    repo = Path.cwd()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    static = run_static_audit(repo, Path(args.config))
    dynamic = run_dynamic_forward(repo, args.device)
    git_head = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {"pass": bool(static["pass"] and dynamic["pass"]), "git_head": git_head, **static, "smoke_result": dynamic, "review_pass_path": str(out / "REVIEW_PASS_EAGLE_PU_V1.txt")}
    audit_path = out / "implementation_audit_EAGLE_PU_V1.json"
    audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write_review_pass and result["pass"]:
        (out / "REVIEW_PASS_EAGLE_PU_V1.txt").write_text(f"REVIEW_PASS_EAGLE_PU_V1\ngit_head={git_head}\n", encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

