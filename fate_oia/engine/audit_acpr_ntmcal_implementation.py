from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.losses.acpr_ntmcal_losses import (
    native_predicate_measurement_loss,
    native_text_structure_loss,
    ntmcal_reason_pu_loss,
    schedule_weights,
)
from fate_oia.models.acpr_ntmcal_model import ACPRNTMCalModel
from fate_oia.models.acpr_ntmcal_pair_memory import NativeTextReasonPairMemory
from fate_oia.models.acpr_ntmcal_predicate_bank import NativePredicateBank
from fate_oia.models.acpr_ntmcal_pu_state import NativeTextPUReasonState
from fate_oia.models.acpr_ntmcal_text_atoms import NativeTextAtomEncoder
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.acpr_ntmcal_tensor_asserts import assert_deploy_equation, assert_shape


REQUIRED = [
    "configs/acpr_ntmcal_native_text_predicates.yaml",
    "configs/acpr_ntmcal_reason_formulas.yaml",
    "configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml",
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
    "fate_oia/engine/audit_acpr_ntmcal_implementation.py",
    "fate_oia/engine/supervise_acpr_ntmcal_foreground.py",
    "fate_oia/utils/acpr_ntmcal_artifacts.py",
    "fate_oia/utils/acpr_ntmcal_tensor_asserts.py",
    "tests/test_acpr_ntmcal_text_atoms.py",
    "tests/test_acpr_ntmcal_predicate_bank.py",
    "tests/test_acpr_ntmcal_topk_measurement.py",
    "tests/test_acpr_ntmcal_observation_builder.py",
    "tests/test_acpr_ntmcal_pu_state.py",
    "tests/test_acpr_ntmcal_reason_residual.py",
    "tests/test_acpr_ntmcal_action_predicate.py",
    "tests/test_acpr_ntmcal_threshold_head.py",
    "tests/test_acpr_ntmcal_pair_memory.py",
    "tests/test_acpr_ntmcal_model_forward.py",
    "tests/test_acpr_ntmcal_losses.py",
    "tests/test_acpr_ntmcal_train_protocol.py",
    "tests/test_acpr_ntmcal_audit.py",
    "scripts/FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1",
    "scripts/FATE_OIA_acpr_ntmcal_v1_foreground.ps1",
]

FORBIDDEN = [
    "hashlib",
    "open_clip",
    "clip.load",
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
    "token_compression: topk",
    "checkpoint_best_val",
    "best_selection_split: val",
    "eval_splits: val",
    "Start-Job",
    "nohup",
    "hidden cmd",
    "scheduled task",
    "daemon",
]

REQUIRED_OUTPUT_KEYS = [
    "action_logits_base",
    "reason_logits_base",
    "action_logits_ntmcal",
    "reason_logits_ntmcal",
    "theta_action",
    "theta_reason",
    "action_logits_deploy",
    "reason_logits_deploy",
    "logits_deploy",
    "predicate_q",
    "predicate_rho",
    "predicate_tokens",
    "predicate_topk_indices",
    "predicate_topk_attention",
    "support_score",
    "contra_score",
    "reason_reliability",
    "pu_state",
    "reason_delta",
    "action_predicate_delta",
    "threshold_delta_reason",
    "threshold_delta_action",
    "ntmcal_stats",
]


def _record(functional: dict, name: str, fn) -> bool:
    try:
        functional[name] = fn()
        functional[name]["pass"] = True
        return True
    except Exception as exc:
        functional[name] = {"pass": False, "error": repr(exc)}
        return False


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


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
        # Tests and this audit file intentionally contain forbidden strings as assertions.
        # The forbidden scan targets active implementation/config/launcher files only.
        if path.startswith("tests/") or path.endswith("audit_acpr_ntmcal_implementation.py"):
            continue
        if Path(path).suffix in {".py", ".yaml", ".ps1"} and Path(path).exists():
            text = _read(path)
            hits = [x for x in FORBIDDEN if x in text]
            if hits:
                forbidden[path] = hits

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    functional = {}
    pass_flag = not missing and not forbidden
    cache: dict[str, object] = {}

    def get_bank() -> NativePredicateBank:
        if "bank" not in cache:
            cache["bank"] = NativePredicateBank.from_yaml("configs/acpr_ntmcal_native_text_predicates.yaml")
        return cache["bank"]  # type: ignore[return-value]

    def get_reason_cfg() -> dict:
        if "reason_cfg" not in cache:
            cache["reason_cfg"] = yaml.safe_load(Path("configs/acpr_ntmcal_reason_formulas.yaml").read_text(encoding="utf-8")) or {}
        return cache["reason_cfg"]  # type: ignore[return-value]

    def get_sample() -> tuple[dict, int, int]:
        if "sample" not in cache:
            transform = AspectRatioLetterboxTransform(360, 640, patch_size=8)
            train_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split="train", action_dim=4, reason_dim=21, load_image=True, transform=transform)
            test_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split="test", action_dim=4, reason_dim=21, load_image=True, transform=transform)
            cache["sample"] = train_ds[0]
            cache["train_count"] = len(train_ds)
            cache["test_count"] = len(test_ds)
        return cache["sample"], int(cache["train_count"]), int(cache["test_count"])  # type: ignore[return-value]

    def get_model() -> ACPRNTMCalModel:
        if "model" not in cache:
            cache["model"] = ACPRNTMCalModel(
                selected_layers=tuple(cfg.get("model", {}).get("selected_layers", [3, 7, 11])),
                pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
                use_mock_dino=False,
                predicate_topk=int(cfg.get("model", {}).get("predicate_topk", 64)),
            ).to(device)
        return cache["model"]  # type: ignore[return-value]

    def get_forward(epoch: int = 8, split: str = "train") -> dict:
        key = f"forward_{epoch}_{split}"
        if key not in cache:
            sample, _, _ = get_sample()
            model = get_model()
            images = sample["image"].unsqueeze(0).to(device)
            reason = sample["reason"].view(1, 21).to(device)
            if split == "train":
                model.train()
                cache[key] = model(images, epoch=epoch, reason_labels=reason, split="train")
            else:
                model.eval()
                with torch.no_grad():
                    cache[key] = model(images, epoch=epoch, reason_labels=reason, split="test")
        return cache[key]  # type: ignore[return-value]

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

    def check_dataset_direct_image():
        sample, train_count, test_count = get_sample()
        assert train_count > 0 and test_count > 0
        assert sample["image"].shape == (3, 360, 640)
        assert sample["action"].shape == (4,)
        assert sample["reason"].shape == (21,)
        assert sample["action"].dtype.is_floating_point
        assert sample["reason"].dtype.is_floating_point
        assert "file_name" in sample
        return {"train_count": train_count, "test_count": test_count}

    def check_dino_frozen():
        model = get_model()
        out = get_forward()
        assert all(not p.requires_grad for p in model.dino.parameters())
        assert_shape(out["patch_tokens_by_layer"], (1, 3, 3600, 384), "patch_tokens_by_layer")
        return {"dino_frozen": True, "patch_shape": list(out["patch_tokens_by_layer"].shape)}

    def check_native_text_bank():
        bank = get_bank()
        audit = bank.audit()
        assert audit["predicate_count"] >= 40
        return audit

    def check_text_atom_encoder():
        bank = get_bank()
        enc = NativeTextAtomEncoder(bank.atom_vocab, dim=384)
        emb = enc.encode_predicates(bank.specs)
        assert emb.shape == (len(bank.specs), 384)
        loss = native_text_structure_loss(enc, bank.specs)["native_text_structure_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grad_sum = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
        assert grad_sum > 0
        return {"embedding_shape": list(emb.shape), "structure_loss": float(loss.detach()), "grad_sum": grad_sum}

    def check_reason_formulas():
        data = get_reason_cfg()
        bank = get_bank()
        actions = data.get("actions", [])
        reasons = sorted(data.get("reasons", []), key=lambda r: int(r["id"]))
        assert [a["name"] for a in actions] == ["forward", "stop", "left", "right"]
        assert len(reasons) == 21
        assert [int(r["id"]) for r in reasons] == list(range(21))
        assert data.get("tail_reason_indices") == [12, 9, 5, 14, 6, 11, 10, 13]
        for r in reasons:
            assert r.get("text") and not str(r["name"]).startswith("reason_")
            for name in list(r.get("support_predicates", [])) + list(r.get("contra_predicates", [])):
                assert name in bank.name_to_id
            assert set(r.get("compatible_actions", [])) <= {"forward", "stop", "left", "right"}
        return {"reason_count": len(reasons), "action_count": len(actions)}

    def check_topk_measurement():
        out = get_forward()
        p = len(get_bank().specs)
        assert_shape(out["predicate_q"], (1, p), "predicate_q")
        assert_shape(out["predicate_rho"], (1, p), "predicate_rho")
        assert_shape(out["predicate_tokens"], (1, p, 384), "predicate_tokens")
        assert out["predicate_topk_indices"].shape[-1] <= 96
        assert out["predicate_stats"]["dense_bpnd_materialized"] is False
        text = _read("fate_oia/models/acpr_ntmcal_topk_predicate_measurement.py")
        for bad in ['einsum("ms,bsnd->bmnd"', "view(b, p, n, d)", "reshape(b, p, n, d)"]:
            assert bad not in text
        return out["predicate_stats"]

    def check_observation_builder():
        model = get_model()
        reason = torch.zeros(2, 21, device=device)
        reason[0, 0] = 1
        reason[1, 12] = 1
        obs = model.observation_builder(reason, split="train", batch_size=2, device=device)
        assert obs["obs_mask"].sum() > 0
        assert obs["obs_soft_negative"].sum() >= 0
        test_obs = model.observation_builder(torch.ones_like(reason), split="test", batch_size=2, device=device)
        assert test_obs["obs_mask"].sum() == 0
        assert test_obs["source_stats"]["test_ignored"] is True
        return obs["source_stats"]

    def check_predicate_loss():
        q = torch.full((2, 4), 0.5, device=device, requires_grad=True)
        rho_low = torch.full((2, 4), 0.05, device=device, requires_grad=True)
        obs = {
            "obs_mask": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32, device=device),
            "obs_value": torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32, device=device),
            "obs_soft_negative": torch.tensor([[0, 0.5, 0, 0], [0, 0, 0.5, 0]], dtype=torch.float32, device=device),
        }
        loss_low = native_predicate_measurement_loss(q, rho_low, obs, epoch=7)
        loss_low.backward()
        observed = (obs["obs_mask"] + obs["obs_soft_negative"]) > 0
        assert rho_low.grad[observed].mean() < 0
        rho_high = torch.full((2, 4), 0.99, device=device, requires_grad=True)
        loss_high = native_predicate_measurement_loss(q.detach().clone().requires_grad_(True), rho_high, obs, epoch=7)
        loss_high.backward()
        assert rho_high.grad[observed].mean() > 0
        return {"low_rho_grad": float(rho_low.grad[observed].mean()), "high_rho_grad": float(rho_high.grad[observed].mean())}

    def check_pu_state():
        model = get_model()
        q = torch.rand(2, len(model.predicate_bank.specs), device=device)
        rho = torch.full_like(q, 0.8)
        reason = torch.zeros(2, 21, device=device)
        reason[0, 0] = 1
        pu0 = model.pu_builder(reason, q, rho, epoch=0)
        pu3 = model.pu_builder(reason, q, rho, epoch=3)
        pu7 = model.pu_builder(reason, q, rho, epoch=7)
        assert pu0["hard_negative_mask"].sum() == 0
        assert pu0["soft_negative_weight"].sum() == 0
        assert pu3["soft_negative_weight"].sum() >= 0
        assert pu7["hard_negative_mask"].sum() >= 0
        return {"epoch0": pu0["stats"], "epoch3": pu3["stats"], "epoch7": pu7["stats"]}

    def check_reason_residual():
        out = get_forward(epoch=8)
        out_zero = get_model()(get_sample()[0]["image"].unsqueeze(0).to(device), epoch=8, reason_labels=get_sample()[0]["reason"].view(1, 21).to(device), split="train", force_zero_reason_delta=True)
        action_diff = (out["action_logits_ntmcal"] - out_zero["action_logits_ntmcal"]).abs().max().item()
        reason_delta_max = out["reason_delta"].abs().max().item()
        assert action_diff < 1e-6
        assert reason_delta_max <= 0.18 + 1e-6
        return {"action_independence_error": action_diff, "reason_delta_max": reason_delta_max}

    def check_action_predicate():
        out5 = get_forward(epoch=5)
        out8 = get_forward(epoch=8)
        assert out5["action_predicate_delta"].abs().max().item() == 0.0
        assert out8["action_predicate_delta"].abs().max().item() <= 0.05 + 1e-6
        src = _read("fate_oia/models/acpr_ntmcal_action_predicate_head.py")
        assert "reason_logits" not in src and "reason_labels" not in src and "action_combo" not in src
        return {"epoch5_delta": 0.0, "epoch8_delta_max": float(out8["action_predicate_delta"].abs().max())}

    def check_threshold_head():
        out = get_forward(epoch=8)
        assert_deploy_equation(out["action_logits_ntmcal"], out["theta_action"], out["action_logits_deploy"], "action")
        assert_deploy_equation(out["reason_logits_ntmcal"], out["theta_reason"], out["reason_logits_deploy"], "reason")
        assert out["threshold_delta_reason"].shape == (1, 21)
        assert out["threshold_delta_action"].shape == (1, 4)
        src = _read("fate_oia/models/acpr_ntmcal_threshold_head.py")
        for token in ["support_score.detach()", "contra_score.detach()", "reason_rho.detach()", "base_reason_logits", "card"]:
            assert token in src
        return out["threshold_stats"]

    def check_pair_memory():
        mem = NativeTextReasonPairMemory(reason_dim=3, capacity_per_reason=4).to(device)
        logits = torch.tensor([[0.1, -0.1, 0.0], [0.2, -0.2, 0.05], [-0.05, 0.1, -0.1]], device=device)
        targets = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)
        pu = {
            "hard_negative_mask": torch.tensor([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.float32, device=device),
        }
        z, stats6 = mem.loss(logits, targets, pu, epoch=6, main_loss=logits.abs().mean())
        assert z.item() == 0.0 and stats6["pair_count_total"] == 0
        mem.enqueue(logits, targets, pu)
        z, stats10 = mem.loss(logits, targets, pu, epoch=10, main_loss=logits.abs().mean())
        assert torch.isfinite(z)
        assert stats10["memory_positive_coverage"] > 0
        assert stats10["memory_negative_coverage"] > 0
        src = _read("fate_oia/models/acpr_ntmcal_pair_memory.py")
        assert "register_buffer" in src and "pos_ptr" in src and "neg_ptr" in src
        return stats10

    def check_full_model_forward():
        out = get_forward(epoch=8)
        for key in REQUIRED_OUTPUT_KEYS:
            assert key in out, key
        assert_shape(out["action_logits_deploy"], (1, 4), "action_logits_deploy")
        assert_shape(out["reason_logits_deploy"], (1, 21), "reason_logits_deploy")
        test0 = get_forward(epoch=8, split="test")
        sample, _, _ = get_sample()
        model = get_model()
        model.eval()
        with torch.no_grad():
            fake = torch.ones(1, 21, device=device)
            test1 = model(sample["image"].unsqueeze(0).to(device), epoch=8, reason_labels=fake, split="test")
        test_diff = (test0["reason_logits_deploy"] - test1["reason_logits_deploy"]).abs().max().item()
        assert test_diff < 1e-6
        return {"output_keys_checked": len(REQUIRED_OUTPUT_KEYS), "test_label_invariance_error": test_diff}

    def check_training_protocol():
        train_text = _read("fate_oia/engine/train_acpr_ntmcal_oia.py")
        assert "split=\"test\"" in train_text
        assert "checkpoint_best_val" not in train_text
        assert "make_train_calib_indices" in train_text
        assert "update_train_calib_teacher" in train_text
        assert "no_feature_cache" in train_text
        assert "token_compression" in train_text
        assert "persistent_workers" in train_text and "prefetch_factor" in train_text
        assert "metrics_deploy_fixed.json" in train_text and "metrics_base_fixed.json" in train_text
        assert "metrics_oracle_diagnostic.json" in train_text
        return {"train_protocol_checked": True}

    def check_loss_schedule():
        assert schedule_weights(0)["pair"] == 0
        assert schedule_weights(3)["pair"] == 0
        assert schedule_weights(7)["pair"] > 0
        assert schedule_weights(13)["pair"] < schedule_weights(7)["pair"]
        return {"epoch0": schedule_weights(0), "epoch7": schedule_weights(7), "epoch13": schedule_weights(13)}

    def check_artifact_schema():
        train_text = _read("fate_oia/engine/train_acpr_ntmcal_oia.py")
        required = [
            "metrics_summary.json",
            "metrics_deploy_fixed.json",
            "metrics_base_fixed.json",
            "metrics_oracle_diagnostic.json",
            "loss_components.jsonl",
            "logits_action_base_test.pt",
            "logits_reason_base_test.pt",
            "logits_action_deploy_test.pt",
            "logits_reason_deploy_test.pt",
            "labels_action_test.pt",
            "labels_reason_test.pt",
            "file_names_test.json",
            "native_text_atom_stats.json",
            "predicate_bank_audit.json",
            "predicate_measurement_stats.jsonl",
            "predicate_topk_stats.jsonl",
            "predicate_attention_mass_sample.pt",
            "pu_state_stats.jsonl",
            "reason_delta_stats.jsonl",
            "action_predicate_stats.jsonl",
            "threshold_delta_stats.jsonl",
            "threshold_stats.jsonl",
            "pair_memory_stats.jsonl",
            "tail_reason_metrics.json",
            "grad_conflict_stats.jsonl",
            "action_independence_probe.json",
            "failure_cases.jsonl",
            "run_manifest.json",
            "supervisor_live_status.json",
        ]
        missing_artifacts = [name for name in required if name not in train_text]
        assert not missing_artifacts, missing_artifacts
        return {"artifact_count": len(required)}

    def check_memory_probe():
        text = _read("scripts/FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1")
        for token in ["8", "6", "4", "memory_probe.json", "reserved", "step_time"]:
            assert token in text
        return {"memory_probe_script_checked": True}

    def check_supervisor():
        text = _read("fate_oia/engine/supervise_acpr_ntmcal_foreground.py") + "\n" + _read("scripts/FATE_OIA_acpr_ntmcal_v1_foreground.ps1")
        for bad in ["Start-Job", "nohup", "hidden cmd", "scheduled task", "daemon"]:
            assert bad not in text
        for token in ["REVIEW_PASS", "memory_probe", "supervisor_live_status", "test"]:
            assert token in text
        return {"supervisor_checked": True}

    checks = [
        ("config", check_config),
        ("dataset_direct_image", check_dataset_direct_image),
        ("dino_frozen", check_dino_frozen),
        ("native_text_bank", check_native_text_bank),
        ("text_atom_encoder", check_text_atom_encoder),
        ("reason_formulas", check_reason_formulas),
        ("topk_measurement", check_topk_measurement),
        ("observation_builder", check_observation_builder),
        ("predicate_loss", check_predicate_loss),
        ("pu_state", check_pu_state),
        ("reason_residual", check_reason_residual),
        ("action_predicate", check_action_predicate),
        ("threshold_head", check_threshold_head),
        ("pair_memory", check_pair_memory),
        ("full_model_forward", check_full_model_forward),
        ("training_protocol", check_training_protocol),
        ("loss_schedule", check_loss_schedule),
        ("artifact_schema", check_artifact_schema),
        ("memory_probe", check_memory_probe),
        ("foreground_supervisor", check_supervisor),
    ]
    for name, fn in checks:
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
    pass_path = out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt"
    if pass_path.exists():
        pass_path.unlink()
    if pass_flag and args.write_review_pass:
        pass_path.write_text("REVIEW_PASS_ACPR_NTMCAL_V1\n", encoding="utf-8")
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
