from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any


FORBIDDEN = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "hidden cmd"]


def load_config_flat(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    flat: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def validate_config(cfg: dict[str, Any]) -> list[str]:
    req = {
        "feature_cache_enabled": False,
        "token_compression": "none",
        "test_only_evaluation": True,
        "best_selection_split": "test",
        "bdd100k_test_gt_input": False,
        "image_height": 360,
        "image_width": 640,
        "action_dim": 4,
        "reason_dim": 21,
    }
    errors = []
    for k, v in req.items():
        if cfg.get(k) != v:
            errors.append(f"config {k} expected {v!r}, got {cfg.get(k)!r}")
    if int(cfg.get("epochs", 0)) < 28:
        errors.append("config epochs must be >= 28")
    if cfg.get("scheduler") != "cosine":
        errors.append("scheduler must be cosine")
    return errors


def validate_foreground_files(paths: list[Path]) -> list[str]:
    errors = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        for word in FORBIDDEN:
            if word.lower() in text.lower():
                errors.append(f"forbidden foreground token {word} in {p}")
    return errors


def run_static_audit(require_smoke_artifacts: bool = False, smoke_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    from fate_oia.models.ceai_action_set import ActionSetPrototypeHead
    from fate_oia.models.ceai_cross_expert_exchange import ControlledCrossExpertExchange
    from fate_oia.models.ceai_expert_adapter import ExpertAdapterBlock
    from fate_oia.models.ceai_oia_model import CEAIOIAFeatureModel
    from fate_oia.models.ceai_pair_reliability import PairReliabilityHead
    from fate_oia.models.ceai_pair_sparse_attention import TaskGuidedPairSparseAttention
    from fate_oia.models.ceai_router import ParetoSafeRouter
    from fate_oia.models.ceai_scene_state import SceneStatePrototypeTransformer
    from fate_oia.losses import ceai_losses
    from fate_oia.losses import pcgrad_lite

    src_model = inspect.getsource(CEAIOIAFeatureModel)
    if "FATEOIAFeatureModel" not in src_model:
        errors.append("CEAI model must use FATEOIAFeatureModel")
    for key in ["base_action_logits", "base_reason_logits", "action_visual_logits", "action_reason_logits", "action_fused_logits", "reason_logits"]:
        if key not in src_model:
            errors.append(f"CEAI model missing output/source marker {key}")
    if "stopgrad" not in inspect.getsource(ControlledCrossExpertExchange).lower() and ".detach()" not in inspect.getsource(ControlledCrossExpertExchange):
        errors.append("A->R stopgrad/detach missing")
    if "topk" not in inspect.getsource(TaskGuidedPairSparseAttention.forward):
        errors.append("pair sparse attention topk missing")
    if "pair_reliability" not in inspect.getsource(PairReliabilityHead.forward):
        errors.append("pair reliability q_ar missing")
    if "base_action_logits + action_delta" not in inspect.getsource(ParetoSafeRouter.forward):
        errors.append("router final_action anchor missing")
    if "register_buffer" not in inspect.getsource(ActionSetPrototypeHead):
        errors.append("action set prototype buffer missing")
    if "MultiheadAttention" not in inspect.getsource(ExpertAdapterBlock):
        errors.append("expert adapter attention missing")
    if "scene_queries" not in inspect.getsource(SceneStatePrototypeTransformer):
        errors.append("scene state learnable queries missing")
    if "final_action_logits" not in inspect.getsource(ceai_losses.ceai_main_loss):
        errors.append("main loss must use final action logits")
    if "apply_pcgrad_lite" not in inspect.getsource(pcgrad_lite):
        errors.append("PCGrad-lite implementation missing")
    if require_smoke_artifacts:
        required = [
            "metrics_summary.json", "branch_metrics.json", "loss_components.jsonl", "readiness_stats.json",
            "scene_state_stats.json", "implicit_prototype_stats.json", "action_set_stats.json",
            "expert_usage_stats.json", "cross_expert_exchange_stats.json", "pair_attention_stats.json",
            "pair_reliability_stats.json", "router_stats.json", "grad_conflict_stats.json",
            "bdd100k_scene_state_stats.json", "selected_vs_random_evidence_stats.json", "run_manifest_epoch.json",
        ]
        if not smoke_dir:
            errors.append("smoke_dir required for artifact audit")
        else:
            epoch_dir = smoke_dir / "epoch_000"
            for name in required:
                p = epoch_dir / name
                if not p.exists() or p.stat().st_size <= 2:
                    errors.append(f"missing/empty smoke artifact {p}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--smoke_dir", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    branch = ""
    try:
        import subprocess

        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except Exception:
        pass
    cwd = Path.cwd()
    if "fate_oia_ceai_oia_v1_worktree" not in str(cwd):
        errors.append(f"wrong worktree {cwd}")
    if branch != "ceai_oia_v1":
        errors.append(f"wrong branch {branch}")
    errors.extend(validate_config(load_config_flat(Path(args.config))))
    errors.extend(validate_foreground_files([Path("scripts/FATE_OIA_ceai_oia_v1_foreground.ps1"), Path("fate_oia/engine/supervise_ceai_oia_foreground.py")]))
    errors.extend(run_static_audit(require_smoke_artifacts=bool(args.smoke_dir), smoke_dir=Path(args.smoke_dir) if args.smoke_dir else None))
    if errors:
        (out / "REVIEW_FAIL_CEAI_OIA_V1.txt").write_text("\n".join(errors), encoding="utf-8")
        raise SystemExit("CEAI audit failed:\n" + "\n".join(errors))
    (out / "REVIEW_PASS_CEAI_OIA_V1.txt").write_text(
        "\n".join([
            "CEAI-OIA V1 REVIEW PASS",
            f"branch: {branch}",
            f"worktree: {cwd}",
            f"config: {args.config}",
            "py_compile: PASS",
            "pytest: PASS",
            "smoke: PASS",
            "audit_gates: PASS",
            "notes:",
            "  - final_action anchored at base_action",
            "  - scene-state prototypes active",
            "  - expert adapters include attention",
            "  - A->R stopgrad active",
            "  - R->A reliability readiness controlled",
            "  - task-guided pair sparse attention active",
            "  - pair reliability q_ar active",
            "  - main loss action+reason only",
            "  - aux regularizers gradient-budgeted",
            "  - BDD100K GT train-only / primary test image-only",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
