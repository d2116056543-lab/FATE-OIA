from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.models.acpr_ntmcal_model import ACPRNTMCalModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.acpr_ntmcal_tensor_asserts import assert_deploy_equation, assert_shape


REQUIRED = [
    "configs/acpr_ntmcal_native_text_predicates.yaml",
    "configs/acpr_ntmcal_reason_formulas.yaml",
    "fate_oia/models/acpr_ntmcal_text_atoms.py",
    "fate_oia/models/acpr_ntmcal_predicate_bank.py",
    "fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py",
    "fate_oia/models/acpr_ntmcal_observation_builder.py",
    "fate_oia/models/acpr_ntmcal_pu_state.py",
    "fate_oia/models/acpr_ntmcal_reason_residual.py",
    "fate_oia/models/acpr_ntmcal_action_predicate_head.py",
    "fate_oia/models/acpr_ntmcal_threshold_head.py",
    "fate_oia/models/acpr_ntmcal_pair_memory.py",
    "fate_oia/models/acpr_ntmcal_model.py",
    "fate_oia/losses/acpr_ntmcal_losses.py",
    "fate_oia/engine/train_acpr_ntmcal_oia.py",
    "fate_oia/engine/eval_acpr_ntmcal_oia.py",
    "fate_oia/engine/supervise_acpr_ntmcal_foreground.py",
    "scripts/FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1",
    "scripts/FATE_OIA_acpr_ntmcal_v1_foreground.ps1",
]
FORBIDDEN = [
    "hashlib",
    "open_clip",
    "CLIPModel",
    "AutoTokenizer",
    "AutoModel",
    "BertModel",
    "SentenceTransformer",
    "sentence_transformers",
    "frozen_run_c",
    "FrozenRunC",
    "run_c_logits",
    "cached_logits",
    "tail_residual_adapter",
    "checkpoint distillation",
    "teacher checkpoint",
    "expert",
    "Expert",
    "moe",
    "MoE",
    "specialist",
    "Specialist",
    "router",
    "Router",
    "pmi",
    "cooccur",
    "co_occurrence",
    "label_correlation",
    "feature_cache_enabled: true",
    "token_compression: keep_merge",
    "checkpoint_best_val",
    "best_selection_split: val",
    "eval_splits: val",
    "Start-Job",
    "nohup",
    "scheduled task",
    "daemon",
]


def _record(functional: dict, name: str, fn) -> bool:
    try:
        functional[name] = fn()
        functional[name]["pass"] = True
        return True
    except Exception as exc:
        functional[name] = {"pass": False, "error": repr(exc)}
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED if not Path(p).exists()]
    forbidden = {}
    for path in REQUIRED:
        if Path(path).suffix in {".py", ".yaml", ".ps1"} and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            hits = [x for x in FORBIDDEN if x in text]
            if hits:
                forbidden[path] = hits
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    functional = {}
    pass_flag = not missing and not forbidden
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    def check_config():
        assert cfg["image_height"] == 360 and cfg["image_width"] == 640 and cfg["patch_size"] == 8
        assert cfg["training"]["best_selection_split"] == "test"
        assert cfg["evaluation"]["splits"] == ["test"]
        assert cfg["training"]["no_feature_cache"] is True
        assert cfg["training"]["token_compression"] == "none"
        assert cfg["model"]["use_mock_dino"] is False
        assert cfg["model"]["dino_no_grad"] is True
        assert int(cfg["model"]["predicate_topk"]) <= 96
        assert cfg["ntmcal"]["teacher_source"] == "train_calib"
        assert cfg["ntmcal"]["oracle_test_thresholds"] == "diagnostic_only"
        assert cfg["pu"]["soft_negative_start_epoch"] == 3
        assert cfg["pu"]["hard_negative_start_epoch"] == 7
        assert cfg["pair"]["start_epoch"] >= 7
        return {"config_checked": True}

    def check_dataset_and_forward():
        transform = AspectRatioLetterboxTransform(360, 640, patch_size=8)
        train_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split="train", action_dim=4, reason_dim=21, load_image=True, transform=transform)
        test_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split="test", action_dim=4, reason_dim=21, load_image=True, transform=transform)
        sample = train_ds[0]
        assert sample["image"].shape == (3, 360, 640)
        assert sample["action"].shape == (4,)
        assert sample["reason"].shape == (21,)
        model = ACPRNTMCalModel(
            selected_layers=tuple(cfg.get("model", {}).get("selected_layers", [3, 7, 11])),
            pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
            use_mock_dino=False,
            predicate_topk=8,
        ).to(device)
        assert all(not p.requires_grad for p in model.dino.parameters())
        images = sample["image"].unsqueeze(0).to(device)
        reason = sample["reason"].view(1, 21).to(device)
        model.train()
        out = model(images, epoch=8, reason_labels=reason, split="train")
        assert_shape(out["patch_tokens_by_layer"], (1, 3, 3600, 384), "patch_tokens_by_layer")
        assert_shape(out["action_logits_deploy"], (1, 4), "action_logits_deploy")
        assert_shape(out["reason_logits_deploy"], (1, 21), "reason_logits_deploy")
        assert_shape(out["predicate_q"], (1, None), "predicate_q")
        assert_shape(out["predicate_rho"], (1, None), "predicate_rho")
        assert_shape(out["predicate_tokens"], (1, None, 384), "predicate_tokens")
        assert out["predicate_topk_indices"].shape[-1] <= 96
        assert out["predicate_stats"]["dense_bpnd_materialized"] is False
        assert_deploy_equation(out["action_logits_ntmcal"], out["theta_action"], out["action_logits_deploy"], "action")
        assert_deploy_equation(out["reason_logits_ntmcal"], out["theta_reason"], out["reason_logits_deploy"], "reason")
        out_zero = model(images, epoch=8, reason_labels=reason, split="train", force_zero_reason_delta=True)
        action_diff = (out["action_logits_ntmcal"] - out_zero["action_logits_ntmcal"]).abs().max().item()
        assert action_diff < 1e-6
        model.eval()
        fake0 = torch.zeros_like(reason)
        fake1 = torch.ones_like(reason)
        test0 = model(images, epoch=8, reason_labels=fake0, split="test")
        test1 = model(images, epoch=8, reason_labels=fake1, split="test")
        test_diff = (test0["reason_logits_deploy"] - test1["reason_logits_deploy"]).abs().max().item()
        assert test_diff < 1e-6
        return {
            "train_count": len(train_ds),
            "test_count": len(test_ds),
            "predicate_count": len(model.predicate_bank.specs),
            "action_independence_error": action_diff,
            "test_label_invariance_error": test_diff,
        }

    def check_static_source():
        topk_text = Path("fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py").read_text(encoding="utf-8")
        assert "expand(-1, p, -1, -1)" not in topk_text
        assert "dense_bpnd_materialized" in topk_text
        train_text = Path("fate_oia/engine/train_acpr_ntmcal_oia.py").read_text(encoding="utf-8")
        assert "checkpoint_best_val" not in train_text
        assert "make_train_calib_indices" in train_text
        assert "update_train_calib_teacher" in train_text
        assert "metrics_deploy_fixed.json" in train_text
        assert "predicate_attention_mass_sample.pt" in train_text
        assert "supervisor_live_status.json" in train_text
        loss_text = Path("fate_oia/losses/acpr_ntmcal_losses.py").read_text(encoding="utf-8")
        assert "theta_action_teacher" in loss_text and "predicted_positive_rate_loss" in loss_text
        return {"static_source_checked": True}

    for name, fn in [
        ("config", check_config),
        ("dataset_direct_image_and_full_model_forward", check_dataset_and_forward),
        ("static_source_contracts", check_static_source),
    ]:
        pass_flag = _record(functional, name, fn) and pass_flag

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    result = {
        "pass": bool(pass_flag),
        "git_head": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "worktree": str(Path.cwd()),
        "checked_files": REQUIRED,
        "forbidden_pattern_results": forbidden,
        "functional_checks": functional,
        "memory_probe_result": {},
        "smoke_result": {},
        "review_pass_path": str(out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt"),
        "missing_items": missing,
        "warnings": [],
    }
    (out_dir / "implementation_audit_ACPR_NTMCAL_V1.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if pass_flag and args.write_review_pass:
        (out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt").write_text("REVIEW_PASS_ACPR_NTMCAL_V1\n", encoding="utf-8")
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
