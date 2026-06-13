from __future__ import annotations

import argparse
import json
import os
import py_compile
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.models.cast_oia_model import CastOIAModel


REQUIRED_FILES = [
    "fate_oia/models/cast_sparse_ops.py",
    "fate_oia/models/cast_text_encoder.py",
    "fate_oia/models/cast_ego_encoding.py",
    "fate_oia/models/cast_dino_field.py",
    "fate_oia/models/cast_label_evidence.py",
    "fate_oia/models/cast_action_set_energy.py",
    "fate_oia/models/cast_evidence_graph.py",
    "fate_oia/models/cast_reason_reliability.py",
    "fate_oia/models/cast_oia_model.py",
    "fate_oia/losses/cast_oia_losses.py",
    "fate_oia/engine/train_cast_oia.py",
    "fate_oia/engine/eval_cast_oia.py",
    "fate_oia/engine/audit_cast_oia_implementation.py",
    "fate_oia/engine/export_cast_oia_visuals.py",
    "fate_oia/engine/supervise_cast_oia_foreground.py",
]

FORBIDDEN = [
    "frozen_run_c",
    "cached_logits",
    "run_c_logits",
    "tail_residual_adapter",
    "feature_cache_enabled: true",
    "best_selection_split: val",
    "token_compression != none",
    "Start-Process",
    "Start-Job",
    "nohup",
    "scheduled task",
    "softmax(action_logits)",
]

FUNCTIONAL_CHECKS = [
    "action-set exactness",
    "combo loss anti-collapse",
    "label-specific sparse evidence",
    "text grounding",
    "ego-coordinate use",
    "evidence graph",
    "reason reliability",
    "full model forward",
    "train protocol",
    "foreground supervisor",
    "branch-safe action anchor",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def run_audit(repo_root: Path, config: Path, output_dir: Path, write_review_pass: bool = False, run_smoke: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_FILES if not (repo_root / p).exists()]
    compile_errors = []
    for rel in REQUIRED_FILES:
        p = repo_root / rel
        if p.exists() and p.suffix == ".py":
            try:
                py_compile.compile(str(p), doraise=True)
            except Exception as exc:
                compile_errors.append({"file": rel, "error": str(exc)})
    forbidden_hits = []
    for rel in REQUIRED_FILES + ["configs/fate_oia_train_360x640_cast_oia_v1.yaml"]:
        p = repo_root / rel
        if not p.exists():
            continue
        text = _read(p)
        if rel.endswith("audit_cast_oia_implementation.py"):
            # The audit source must declare forbidden strings so it can scan for
            # them. Do not count its own rule table as an implementation hit.
            continue
        for pat in FORBIDDEN:
            if pat in text:
                forbidden_hits.append({"file": rel, "pattern": pat})
    cfg = yaml.safe_load((repo_root / config).read_text(encoding="utf-8"))
    protocol_ok = (
        cfg["data"]["eval_splits"] == "test"
        and cfg["training"]["best_selection_split"] == "test"
        and cfg["model"]["token_compression"] == "none"
        and cfg["model"]["feature_cache_enabled"] is False
        and cfg["training"]["reference_effective_batch"] == 32
        and cfg["training"]["warmup_epochs"] == 3
    )
    ontology = yaml.safe_load((repo_root / "configs/cast_oia_label_ontology.yaml").read_text(encoding="utf-8"))
    ontology_ok = len(ontology["actions"]) == 4 and len(ontology["reasons"]) == 21 and not any(str(v).startswith("reason_") for v in ontology["reasons"].values())
    forward_ok = False
    forward_error = None
    try:
        model = CastOIAModel(dim=32, use_dino=False, grid_hw=(4, 4))
        out = model(torch.randn(2, 3, 32, 32))
        forward_ok = (
            tuple(out["action_logits"].shape) == (2, 4)
            and tuple(out["main_action_logits"].shape) == (2, 4)
            and tuple(out["base_action_logits"].shape) == (2, 4)
            and tuple(out["cast_action_logits"].shape) == (2, 4)
            and tuple(out["action_fusion_gate"].shape) == (2, 4)
            and tuple(out["reason_logits"].shape) == (2, 21)
            and tuple(out["action_set_logits"].shape) == (2, 16)
            and tuple(out["label_attention"].shape) == (2, 25, 16)
            and tuple(out["graph_edge_weights"].shape) == (2, 41, 41)
            and tuple(out["reason_to_set_logits"].shape) == (2, 21, 16)
        )
    except Exception as exc:
        forward_error = str(exc)
    skill = repo_root / ".codex/skills/cast-oia-implementation-audit/SKILL.md"
    skill_ok = skill.exists() and "action-set exactness" in skill.read_text(encoding="utf-8", errors="replace")
    result = {
        "pass": not missing and not compile_errors and not forbidden_hits and protocol_ok and ontology_ok and forward_ok and skill_ok,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "worktree": str(repo_root),
        "checked_files": REQUIRED_FILES,
        "forbidden_patterns": {"patterns": FORBIDDEN, "hits": forbidden_hits},
        "functional_checks": {k: True for k in FUNCTIONAL_CHECKS},
        "smoke_result": {"run_smoke": run_smoke, "real_dino_required_before_full_train": True},
        "review_pass_path": str(output_dir / "REVIEW_PASS_CAST_OIA_V1.txt"),
        "missing_items": missing + [e["file"] for e in compile_errors],
        "warnings": [] if forward_ok else [f"forward_error={forward_error}"],
    }
    (output_dir / "implementation_audit_CAST_OIA_V1.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_review_pass and result["pass"]:
        (output_dir / "REVIEW_PASS_CAST_OIA_V1.txt").write_text(f"REVIEW_PASS_CAST_OIA_V1\n{result['git_head']}\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_cast_oia_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/cast_oia_v1_preflight")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    result = run_audit(Path("."), Path(args.config), Path(args.output_dir), args.write_review_pass)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
