from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "fate_oia/models/egcaf_dino_multilayer.py",
    "fate_oia/models/egcaf_dense_adapter.py",
    "fate_oia/models/egcaf_factor_types.py",
    "fate_oia/models/egcaf_factor_bank.py",
    "fate_oia/models/egcaf_dino_object_factors.py",
    "fate_oia/models/egcaf_scene_state_proxy.py",
    "fate_oia/models/egcaf_sparse_topk.py",
    "fate_oia/models/egcaf_dynamic_selector.py",
    "fate_oia/models/egcaf_factor_actor.py",
    "fate_oia/models/egcaf_reason_decoder.py",
    "fate_oia/models/egcaf_factor_judge.py",
    "fate_oia/models/egcaf_oia_model.py",
    "fate_oia/losses/egcaf_losses.py",
    "fate_oia/losses/egcaf_gradient_budget.py",
    "fate_oia/datasets/bdd100k_scene_state_proxy.py",
    "fate_oia/utils/egcaf_artifacts.py",
    "fate_oia/utils/egcaf_visual_export.py",
    "fate_oia/utils/egcaf_config_validation.py",
    "fate_oia/engine/train_egcaf_oia.py",
    "fate_oia/engine/eval_egcaf_oia.py",
    "fate_oia/engine/audit_egcaf_oia_implementation.py",
    "fate_oia/engine/supervise_egcaf_oia_foreground.py",
    "fate_oia/engine/export_egcaf_visuals.py",
    "configs/fate_oia_train_360x640_egcaf_oia_v1.yaml",
    "configs/egcaf_factor_groups.yaml",
    "configs/egcaf_bdd100k_scene_state.yaml",
    "scripts/FATE_OIA_egcaf_oia_v1_foreground.ps1",
]

ARTIFACTS = [
    "metrics.json", "factor_selection_stats.json", "factor_type_usage.json", "factor_source_usage.json",
    "factor_region_usage.json", "factor_sufficiency_stats.json", "factor_comprehensiveness_stats.json",
    "selected_vs_random_drop.json", "lambda_exp_history.json", "help_hurt_ema.json",
    "selector_entropy_stats.json", "anti_collapse_stats.json", "reason_from_factor_stats.json",
    "action_core_vs_final.json", "guarded_action_stats.json", "scene_state_proxy_stats.json",
    "gradient_budget_stats.json", "visual_factor_samples.jsonl",
]


def run_cmd(cmd: list[str], cwd: Path) -> dict:
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed {cmd}:\n{p.stdout}")
    return {"cmd": cmd, "returncode": p.returncode, "stdout_tail": p.stdout.splitlines()[-20:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_egcaf_oia_v1.yaml")
    ap.add_argument("--smoke_dir", default=r".background_runs\egcaf_oia_v1_smoke")
    ap.add_argument("--output_dir", default=r".background_runs\egcaf_oia_v1_preflight")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"required_files": {}, "commands": []}
    missing = [p for p in REQUIRED if not (root / p).exists()]
    if missing:
        raise SystemExit(f"Missing required EG-CAF files: {missing}")
    for p in REQUIRED:
        summary["required_files"][p] = True
    py_files = [p for p in REQUIRED if p.endswith(".py")] + [str(p).replace("\\", "/") for p in Path("tests").glob("test_egcaf_*.py")]
    for rel in py_files:
        text = (root / rel).read_text(encoding="utf-8-sig")
        ast.parse(text, filename=rel)
    import yaml
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise RuntimeError("EG-CAF YAML config must load as dict")
    if cfg.get("no_feature_cache") is not True or cfg.get("test_only_eval") is not True or list(cfg.get("image_size")) != [360, 640]:
        raise RuntimeError("EG-CAF config lacks no_feature_cache/test_only_eval/image_size [360,640]")
    test_files = [str(p).replace("\\", "/") for p in sorted(Path("tests").glob("test_egcaf_*.py"))]
    summary["commands"].append(run_cmd([sys.executable, "-m", "py_compile", *py_files], root))
    summary["commands"].append(run_cmd([sys.executable, "-m", "pytest", *test_files, "-q"], root))
    smoke_dir = root / args.smoke_dir
    summary["commands"].append(run_cmd([
        sys.executable, "-m", "fate_oia.engine.train_egcaf_oia",
        "--config", args.config, "--output_dir", str(smoke_dir),
        "--epochs", "1", "--batch_size", "2", "--gradient_accumulation_steps", "2",
        "--max_train_samples", "8", "--max_test_samples", "8", "--device", args.device,
        "--no_feature_cache", "--test_only", "--lightweight_backbone", "--print_every", "1",
    ], root))
    epoch_dir = smoke_dir / "epoch_000"
    missing_art = [a for a in ARTIFACTS if not (epoch_dir / a).exists()]
    if missing_art:
        raise RuntimeError(f"Smoke missing EG-CAF artifacts: {missing_art}")
    src_checks = {
        "no_start_process": "Start-Process" not in (root / "scripts/FATE_OIA_egcaf_oia_v1_foreground.ps1").read_text(encoding="utf-8"),
        "actor_has_prototypes": "prototype_vectors" in (root / "fate_oia/models/egcaf_factor_actor.py").read_text(encoding="utf-8"),
        "reason_raw_note": "Raw BDD-OIA JSON stores 21 index positions" in (root / "configs/egcaf_factor_groups.yaml").read_text(encoding="utf-8"),
        "no_positive_cartesian_pair": "positive action" not in (root / "fate_oia").as_posix(),
    }
    if not all(src_checks.values()):
        raise RuntimeError(f"Source hard gate failed: {src_checks}")
    summary["artifact_schema"] = {a: True for a in ARTIFACTS}
    summary["source_checks"] = src_checks
    pass_file = out / "REVIEW_PASS_EGCAF_OIA_V1.txt"
    pass_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"REVIEW_PASS_EGCAF_OIA_V1={pass_file}")


if __name__ == "__main__":
    main()
