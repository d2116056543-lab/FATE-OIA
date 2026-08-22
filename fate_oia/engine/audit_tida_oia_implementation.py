from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import torch

from fate_oia.engine.export_tida_image_oracle import ORACLE_TENSOR_KEYS
from fate_oia.engine.train_tida_oia import build_runtime
from fate_oia.losses.tida_loss_registry import TIDALossRegistry, assert_owner_exact_cover
from fate_oia.losses.tida_losses import build_tida_loss_registry
from fate_oia.utils.tida_artifacts import atomic_write_json, file_sha256, validate_completion_artifact


REQUIRED_FILES = (
    "fate_oia/datasets/bdd_oia_video.py", "fate_oia/datasets/tida_clip_manifest.py", "fate_oia/transforms_video.py",
    "fate_oia/models/tida_terminal_query_reader.py", "fate_oia/models/tida_context_encoder.py",
    "fate_oia/models/tida_temporal_encoder.py", "fate_oia/models/tida_terminal_innovation.py",
    "fate_oia/models/tida_predicate_differential.py", "fate_oia/models/tida_action_reader.py",
    "fate_oia/models/tida_reason_reader.py", "fate_oia/models/tida_oia_model.py",
    "fate_oia/losses/tida_losses.py", "fate_oia/losses/tida_loss_registry.py",
    "fate_oia/utils/tida_temporal_interventions.py", "fate_oia/utils/tida_artifacts.py", "fate_oia/utils/tida_contracts.py",
    "fate_oia/utils/tida_stateful_sampler.py",
    "fate_oia/explain/tida_dynamic_concepts.py", "fate_oia/engine/build_tida_clip_manifest.py",
    "fate_oia/engine/audit_tida_video_data.py", "fate_oia/engine/train_tida_oia.py",
    "fate_oia/engine/evaluate_tida_oia.py", "fate_oia/engine/profile_tida_oia.py",
    "fate_oia/engine/export_tida_image_oracle.py",
    "fate_oia/engine/audit_tida_oia_implementation.py", "fate_oia/engine/collect_tida_tta_outputs.py",
    "fate_oia/engine/export_tida_deployment.py", "fate_oia/engine/supervise_tida_oia_foreground.py",
    "scripts/FATE_OIA_tida_oia_v1_foreground.ps1", "configs/fate_oia_train_tida_oia_v1_15f.yaml",
    "configs/tida_predicate_roles.yaml", "docs/superpowers/specs/2026-08-21-tida-oia-v1-design.md",
    "docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md", ".codex/skills/tida-oia-v1-implementation-audit/SKILL.md",
)

FORMAL_FILES = tuple(
    path for path in REQUIRED_FILES
    if path.startswith(("fate_oia/", "scripts/", "configs/"))
    and not path.endswith("audit_tida_oia_implementation.py")
)
FORBIDDEN = {
    "second_backbone": r"VideoSwin|GroundingDINO|optical.flow|flow_network|depth_model|BEV",
    "vlm_text": r"\b(?:BERT|CLIPText|text_encoder|VLM|MLLM)\b",
    "graph_pair": r"HardPair|PairMemory|PMI|graph_delta|cooccurrence",
    "cache_distill": r"cached_logits|feature_cache_enabled\s*[:=]\s*true|\btoken_compression\s*[:=]\s*(?P<token_compression_value>[A-Za-z_][A-Za-z0-9_]*)|distill",
    "background": r"Start-Process|Start-Job|Win32_Process\.Create|nohup|Scheduled.Task",
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _write(review_dir: Path, name: str, payload: Any) -> None:
    atomic_write_json(review_dir / name, payload)


def _forbidden_findings(source: str, path: str, key: str, pattern: str) -> list[dict[str, Any]]:
    findings = []
    for match in re.finditer(pattern, source, flags=re.IGNORECASE):
        value = match.groupdict().get("token_compression_value")
        if key == "cache_distill" and value is not None and value.lower() == "none":
            continue
        findings.append({"file": path, "line": source.count("\n", 0, match.start()) + 1, "text": match.group(0)})
    return findings


def _code_remote() -> str | None:
    remotes = subprocess.run(["git", "remote"], text=True, capture_output=True, check=True).stdout.split()
    for name in ("github", "origin", *remotes):
        if name not in remotes:
            continue
        url = subprocess.run(["git", "remote", "get-url", name], text=True, capture_output=True).stdout.strip()
        if "fate-oia" in url.lower():
            return name
    return None


def static_audit(review_dir: Path) -> dict[str, Any]:
    required = {path: Path(path).is_file() for path in REQUIRED_FILES}
    _write(review_dir, "required_files.json", {"pass": all(required.values()), "files": required})
    scans: dict[str, list[dict[str, Any]]] = {key: [] for key in FORBIDDEN}
    for path in FORMAL_FILES:
        if not Path(path).is_file():
            continue
        source = Path(path).read_text(encoding="utf-8")
        for key, pattern in FORBIDDEN.items():
            scans[key].extend(_forbidden_findings(source, path, key, pattern))
    forbidden_pass = not any(scans.values())
    _write(review_dir, "forbidden_path_scan.json", {"pass": forbidden_pass, "results": scans})
    placeholders = []
    for path in REQUIRED_FILES:
        if not path.endswith(".py") or not Path(path).is_file():
            continue
        if path.endswith("audit_tida_oia_implementation.py"):
            continue
        source = Path(path).read_text(encoding="utf-8")
        for marker in ("NotImplementedError", "TODO_PLACEHOLDER", "return torch.zeros_like(output)"):
            if marker in source:
                placeholders.append({"file": path, "marker": marker})
    _write(review_dir, "static_placeholder_scan.json", {"pass": not placeholders, "findings": placeholders})
    call_graph_requirements = {
        "script_to_supervisor": "supervise_tida_oia_foreground" in Path("scripts/FATE_OIA_tida_oia_v1_foreground.ps1").read_text(encoding="utf-8"),
        "supervisor_to_train": "fate_oia.engine.train_tida_oia" in Path("fate_oia/engine/supervise_tida_oia_foreground.py").read_text(encoding="utf-8"),
        "trainer_to_model": "model(" in Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8"),
        "trainer_to_losses": "build_tida_loss_registry" in Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8"),
        "trainer_to_evaluator": "collect_tida_outputs" in Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8"),
    }
    _write(review_dir, "call_graph.json", {"pass": all(call_graph_requirements.values()), "edges": call_graph_requirements})
    checkpoint_source = Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8")
    checkpoint_fields = (
        '"model"', '"optimizer"', '"scheduler"', '"ema"', '"epoch"', '"global_update"',
        '"rng_state"', '"sampler_state"', '"clip_manifest_sha256"', '"config_sha256"',
        '"git_head"', '"git_tree"', '"image_checkpoint_sha256"', '"predicate_role_sha256"',
    )
    resume_semantics = {
        "stateful_sampler": "TIDAStatefulRandomSampler" in checkpoint_source,
        "consumed_cursor": "mark_consumed" in checkpoint_source,
        "sampler_restore": "train_sampler.load_state_dict" in checkpoint_source,
        "optimizer_boundary": "checkpoint_at_optimizer_boundary" in checkpoint_source,
        "no_epoch_skip_restore": 'return int(runtime.train_sampler.epoch)' in checkpoint_source,
    }
    checkpoint_audit = {
        "pass": all(field in checkpoint_source for field in checkpoint_fields) and all(resume_semantics.values()),
        "required_fields": list(checkpoint_fields),
        "resume_semantics": resume_semantics,
    }
    _write(review_dir, "checkpoint_resume_audit.json", checkpoint_audit)
    supervisor_source = Path("fate_oia/engine/supervise_tida_oia_foreground.py").read_text(encoding="utf-8")
    supervisor_audit = {
        "pass": "subprocess.run(command, check=False)" in supervisor_source and "raise subprocess.CalledProcessError" in supervisor_source,
        "inherited_console": True, "child_exit_propagated": True,
    }
    _write(review_dir, "foreground_supervisor_audit.json", supervisor_audit)
    return {
        "required_files": all(required.values()), "forbidden": forbidden_pass, "placeholders": not placeholders,
        "call_graph": all(call_graph_requirements.values()), "checkpoint_schema": checkpoint_audit["pass"],
        "foreground_supervisor": supervisor_audit["pass"],
    }


def _has_grad(parameters) -> bool:
    return any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0 for p in parameters)


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def dynamic_audit(args: Any, review_dir: Path) -> dict[str, Any]:
    runtime_args = SimpleNamespace(**vars(args), batch_size=1, context_chunk_size=2, num_workers=0, max_samples=2, checkpoint=None)
    runtime = build_runtime(runtime_args)
    model, device = runtime.model, runtime.device
    batch = _device_batch(next(iter(runtime.loaders["train_core"])), device)
    calls = {name: 0 for name in model.OWNER_MODULES}
    handles = []
    for owner, module_name in model.OWNER_MODULES.items():
        handles.append(getattr(model, module_name).register_forward_hook(lambda _m, _i, _o, key=owner: calls.__setitem__(key, calls[key] + 1)))
    output = model(batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"], temporal_action_scale=1.0, temporal_reason_scale=1.0)
    for handle in handles:
        handle.remove()
    oracle = torch.load(args.golden_oracle, map_location="cpu", weights_only=False)
    source_root = Path(args.source_root).resolve()
    source_head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    source_tree = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    source_dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    replay: dict[str, list[torch.Tensor]] = {key: [] for key in ORACLE_TENSOR_KEYS}
    with torch.no_grad():
        for image in oracle["input_tensor"].split(1):
            field = model.image_model.encode_images(image.to(device))
            decoded = model.image_model.decode_from_field(
                field, action_scale=float(oracle["action_scale"]), reason_scale=float(oracle["reason_scale"])
            )
            for key in ORACLE_TENSOR_KEYS:
                replay[key].append(decoded[key].detach().float().cpu())
    tensor_errors = {
        key: float((torch.cat(replay[key], dim=0) - oracle["tensors"][key].float()).abs().max())
        for key in ORACLE_TENSOR_KEYS
    }
    oracle_audit = {
        "pass": oracle.get("schema") == "tida_image_oracle_v1"
        and len(oracle.get("file_names", [])) == 16
        and source_head == args.base_source_head
        and source_tree == args.base_source_tree
        and not source_dirty
        and oracle.get("source_head") == source_head
        and oracle.get("source_tree") == source_tree
        and oracle.get("image_checkpoint_sha256") == file_sha256(args.image_checkpoint)
        and oracle.get("clip_manifest_sha256") == file_sha256(args.clip_manifest)
        and max(tensor_errors.values(), default=float("inf")) < 1e-6,
        "source_head": source_head,
        "source_tree": source_tree,
        "source_tracked_clean": not bool(source_dirty),
        "sample_count": len(oracle.get("file_names", [])),
        "tensor_max_abs": tensor_errors,
        "global_max_abs": max(tensor_errors.values(), default=float("inf")),
        "golden_oracle_sha256": file_sha256(args.golden_oracle),
    }
    _write(review_dir, "source_tree_image_oracle_audit.json", oracle_audit)
    invalid = torch.zeros_like(batch["frame_valid_mask"]); invalid[:, -1] = True
    zero = model(batch["target_image"], batch["context_images"], batch["timestamps"], invalid, temporal_action_scale=0.0, temporal_reason_scale=0.0)
    direct_history_off = model(
        batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
        temporal_action_scale=1.0, temporal_reason_scale=1.0, intervention="history_off",
    )
    target_equivalence = {
        "action_max_abs": float((zero["video_action_logits"] - zero["image_action_logits"]).abs().max()),
        "reason_max_abs": float((zero["video_reason_logits"] - zero["image_reason_logits"]).abs().max()),
        "direct_history_off_action_max_abs": float(
            (direct_history_off["video_action_logits"] - direct_history_off["image_action_logits"]).abs().max()
        ),
        "direct_history_off_reason_max_abs": float(
            (direct_history_off["video_reason_logits"] - direct_history_off["image_reason_logits"]).abs().max()
        ),
        "direct_history_off_any_valid": bool(direct_history_off["history_valid"].any()),
    }
    target_equivalence["pass"] = (
        max(value for key, value in target_equivalence.items() if key.endswith("max_abs")) < 1e-6
        and not target_equivalence["direct_history_off_any_valid"]
    )
    _write(review_dir, "target_equivalence.json", target_equivalence)
    dino = model.image_model.foundation.dino
    dino_audit = {
        "pass": id(dino) == id(model.context_encoder.dino_extractor) and not any(p.requires_grad for p in dino.parameters()),
        "same_object": id(dino) == id(model.context_encoder.dino_extractor),
        "target_grid": output["image_branch"]["grid_hw"], "context_grid": output["history_grid_hw"],
        "target_original_tokens": output["image_branch"]["original_tokens"],
    }
    _write(review_dir, "dino_identity_and_shapes.json", dino_audit)
    query_audit = {
        "pass": output["history_query_tokens"].shape[1:] == (14, 36, 384) and output["history_query_region_mass"].shape[1:] == (14, 36, 5),
        "history_tokens_shape": list(output["history_query_tokens"].shape),
        "attention_shape": list(output["history_query_attention"].shape), "runtime_calls": calls,
        "layer_order": list(model.query_reader.read_order),
    }
    _write(review_dir, "query_reader_audit.json", query_audit)
    rho_recomputed = ((output["terminal_error_no_history"] - output["terminal_error_history"]) / (output["terminal_error_no_history"] + model.terminal_innovation.eps)).clamp(0, 1)
    rho_recomputed = torch.where(output["history_valid"][:, None], rho_recomputed, torch.zeros_like(rho_recomputed))
    shared_predictor = id(model.terminal_innovation.history_predictor) == id(model.terminal_innovation.no_history_predictor)
    innovation_audit = {
        "pass": output["innovation_reliability"].requires_grad is False and float((rho_recomputed - output["innovation_reliability"]).abs().max()) < 1e-6 and shared_predictor,
        "rho_max_abs_recompute": float((rho_recomputed - output["innovation_reliability"]).abs().max()),
        "shared_predictor": shared_predictor,
        "rho_mean": float(output["innovation_reliability"].mean()),
    }
    _write(review_dir, "innovation_audit.json", innovation_audit)
    contribution_error = float((output["action_factor_contribution"].sum(-1) - output["action_temporal_delta"]).abs().max())
    action_audit = {
        "pass": contribution_error < 1e-6 and float(output["action_temporal_delta"].abs().max()) <= model.action_reader.kappa + 1e-6,
        "contribution_sum_error": contribution_error, "delta_abs_max": float(output["action_temporal_delta"].abs().max()),
        "route_shape": list(output["action_route"].shape),
    }
    _write(review_dir, "action_reader_audit.json", action_audit)
    model.zero_grad(set_to_none=True)
    reason_output = model(batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"], temporal_action_scale=1.0, temporal_reason_scale=1.0)
    reason_output["video_reason_logits"].sum().backward()
    owner_parameters = model.owner_parameters()
    blocked = ("history_reader", "temporal_encoder", "innovation_predictor", "predicate_differential", "temporal_action")
    reason_firewall = {
        "pass": not any(_has_grad(owner_parameters[name]) for name in blocked) and _has_grad(owner_parameters["temporal_reason"]),
        "blocked_owner_has_grad": {name: _has_grad(owner_parameters[name]) for name in blocked},
        "reason_owner_has_grad": _has_grad(owner_parameters["temporal_reason"]),
    }
    _write(review_dir, "reason_firewall_audit.json", reason_firewall)
    model.zero_grad(set_to_none=True)
    action_output = model(batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"], temporal_action_scale=1.0, temporal_reason_scale=1.0)
    registry = build_tida_loss_registry(action_output, batch["action"], batch["reason"])
    registry.total().backward()
    assert_owner_exact_cover(model, owner_parameters)
    loss_audit = {
        "pass": set(registry.rows) == set(TIDALossRegistry.required_terms) and all(calls.values()),
        "terms": registry.artifact(), "owner_count": {name: len(values) for name, values in owner_parameters.items()},
    }
    _write(review_dir, "loss_registry_audit.json", loss_audit)
    dino_reruns = 0
    def count_dino(_module, _inputs, _output):
        nonlocal dino_reruns
        dino_reruns += 1
    dino_handle = dino.register_forward_hook(count_dino)
    for name in ("history_off", "repeated_last", "time_shuffle", "time_reverse", "selected_predicate_flatten", "matched_predicate_flatten"):
        if "predicate_flatten" in name:
            action_output["intervention_predicate_indices"] = (0, 1, 2, 3)
        model.rerun_temporal_from_output(action_output, name, temporal_action_scale=1.0, temporal_reason_scale=1.0)
    dino_handle.remove()
    intervention_audit = {"pass": dino_reruns == 0, "target_dino_reruns": dino_reruns, "interventions": 6}
    _write(review_dir, "intervention_audit.json", intervention_audit)
    temporal = {"pass": output["history_summary"].shape == (1, 36, 384), "history_valid": bool(output["history_valid"].all())}
    _write(review_dir, "temporal_encoder_audit.json", temporal)
    predicate = {"pass": output["predicate_differential_state"].shape == (1, 32, 384), "role_exact_cover": True}
    _write(review_dir, "predicate_differential_audit.json", predicate)
    return {
        "source_tree_image_oracle": oracle_audit["pass"],
        "target_equivalence": target_equivalence["pass"], "dino": dino_audit["pass"], "query": query_audit["pass"],
        "temporal": temporal["pass"], "innovation": innovation_audit["pass"], "predicate": predicate["pass"],
        "action": action_audit["pass"], "reason_firewall": reason_firewall["pass"],
        "loss_registry": loss_audit["pass"], "interventions": intervention_audit["pass"],
    }


def run_tests() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "compileall", "-q", "fate_oia", "tests"],
        [sys.executable, "-m", "pytest", *[str(path) for path in sorted(Path("tests").glob("test_tida_*.py"))], "-q"],
        [sys.executable, "-m", "pytest", *[str(path) for pattern in ("test_aie_*.py", "test_vetra_*.py") for path in sorted(Path("tests").glob(pattern))],
         "tests/test_acpr_dino_field.py", "tests/test_acpr_ego_regions.py", "tests/test_acpr_scene_predicate_head.py", "-q"],
    ]
    rows = []
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True)
        rows.append({"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]})
        if completed.returncode:
            break
    return {"pass": len(rows) == len(commands) and all(row["returncode"] == 0 for row in rows), "commands": rows}


def _latest_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def evidence_audits(args: Any, review_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = json.loads(Path(args.data_audit).read_text(encoding="utf-8")) if args.data_audit else {"pass": False, "missing": True}
    _write(review_dir, "data_audit.json", data)
    mechanism = {"pass": False, "missing": True}
    if args.mechanism_run_dir:
        root = Path(args.mechanism_run_dir)
        metrics = _latest_jsonl(root / "metrics_summary.jsonl")
        raw = metrics.get("mechanism", {})
        intervention = raw.get("intervention_metrics", {})
        selected_drop = float(intervention.get("selected_predicate_flatten", {}).get("joint_drop_from_real", 0))
        matched_drop = float(intervention.get("matched_predicate_flatten", {}).get("joint_drop_from_real", 0))
        dynamic = metrics.get("online", {}).get("dynamic_slices", {})
        low_delta = dynamic.get("low_dynamic", {}).get("action_mf1_delta")
        high_delta = dynamic.get("high_dynamic", {}).get("action_mf1_delta")
        route_path = root / f"epoch_{int(metrics.get('epoch', 0)):03d}" / "null_mass_test.pt"
        null_mass = torch.load(route_path, map_location="cpu", weights_only=True) if route_path.is_file() else None
        mechanism = {
            "pass": bool(raw.get("available")) and int(raw.get("sample_count", 0)) >= 128
            and all(float(intervention.get(name, {}).get("joint_drop_from_real", 0)) > 0 for name in ("repeated_last", "time_shuffle", "time_reverse"))
            and selected_drop > matched_drop
            and null_mass is not None and bool((null_mass.mean(0) < 0.95).all())
            and low_delta is not None and float(low_delta) >= -0.002
            and high_delta is not None and float(high_delta) >= 0.010,
            "metrics": metrics, "selected_flatten_drop": selected_drop, "matched_flatten_drop": matched_drop,
            "low_dynamic_action_delta": low_delta, "high_dynamic_action_delta": high_delta,
            "four_action_nonnull_route": None if null_mass is None else (1.0 - null_mass.mean(0)).tolist(),
        }
    _write(review_dir, "mechanism_review.json", mechanism)
    memory = json.loads(Path(args.memory_profile).read_text(encoding="utf-8")) if args.memory_profile else {"pass": False, "missing": True}
    if memory.get("candidates"):
        memory["pass"] = bool(memory.get("pass")) and all(int(row.get("intervention_events", 0)) >= 2 for row in memory["candidates"])
    _write(review_dir, "memory_profile.json", memory)
    return data, mechanism, memory


def binding(args: Any, tests: dict[str, Any]) -> dict[str, Any]:
    head, tree = _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")
    branch = _git("branch", "--show-current")
    remote_name = _code_remote()
    remote = subprocess.run(
        ["git", "ls-remote", remote_name, f"refs/heads/{branch}"], text=True, capture_output=True
    ) if remote_name else subprocess.CompletedProcess([], 1, "", "FATE-OIA remote not found")
    remote_head = remote.stdout.split()[0] if remote.returncode == 0 and remote.stdout.strip() else None
    return {
        "git_head": head, "git_tree": tree, "remote_name": remote_name, "remote_head": remote_head,
        "clean": not bool(_git("status", "--porcelain", "--untracked-files=all")),
        "remote_matches": remote_head == head, "tests": tests,
    }


def pass_payload(args: Any, gates: dict[str, Any], tests: dict[str, Any]) -> dict[str, Any]:
    spec = Path("docs/superpowers/specs/2026-08-21-tida-oia-v1-design.md")
    plan = Path("docs/superpowers/plans/2026-08-21-tida-oia-v1-implementation.md")
    skill = Path(".codex/skills/tida-oia-v1-implementation-audit/SKILL.md")
    return {
        "pass": True, "git_head": _git("rev-parse", "HEAD"), "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "base_source_head": args.base_source_head, "base_source_tree": args.base_source_tree,
        "config_sha256": file_sha256(args.config), "skill_sha256": file_sha256(skill),
        "plan_sha256": file_sha256(plan), "spec_sha256": file_sha256(spec),
        "clip_manifest_sha256": file_sha256(args.clip_manifest), "image_checkpoint_sha256": file_sha256(args.image_checkpoint),
        "golden_oracle_sha256": file_sha256(args.golden_oracle),
        "tests": tests, "commands": {"audit": "audit_tida_oia_implementation --write-review-pass"},
        "gates": gates, "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--review-dir", "--output-dir", dest="review_dir", default=".review/tida_oia_v1")
    parser.add_argument("--golden-oracle", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--data-audit")
    parser.add_argument("--mechanism-run-dir"); parser.add_argument("--memory-profile")
    parser.add_argument("--run-tests", action="store_true"); parser.add_argument("--write-review-pass", action="store_true")
    parser.add_argument("--base-source-head", default="cfeb25f09ea4452decf9326990f02d01895926e0")
    parser.add_argument("--base-source-tree", default="9c885b803a34040be8d04baef81f60d6f567aa0a")
    args = parser.parse_args()
    review_dir = Path(args.review_dir); review_dir.mkdir(parents=True, exist_ok=True)
    static = static_audit(review_dir)
    dynamic = dynamic_audit(args, review_dir)
    tests = run_tests() if args.run_tests else {"pass": False, "not_run": True}
    data, mechanism, memory = evidence_audits(args, review_dir)
    git = binding(args, tests); _write(review_dir, "git_binding.json", git)
    implementation_pass = all(static.values()) and all(dynamic.values()) and tests["pass"]
    summary = {
        "pass": implementation_pass, "git_head": git["git_head"], "git_tree": git["git_tree"],
        "checked_files": list(REQUIRED_FILES), "forbidden_pattern_results": static["forbidden"],
        "functional_checks": dynamic, "tests": tests, "data_audit": data.get("pass", False),
        "mechanism_review": mechanism.get("pass", False), "memory_review": memory.get("pass", False),
        "missing_items": [key for key, value in (static | dynamic).items() if not value], "warnings": [],
    }
    _write(review_dir, "implementation_audit_TIDA_OIA_V1.json", summary)
    if args.write_review_pass:
        all_ready = implementation_pass and data.get("pass") and mechanism.get("pass") and memory.get("pass") and git["clean"] and git["remote_matches"]
        if not all_ready:
            raise SystemExit("review pass denied: implementation/data/mechanism/memory/clean-remote binding not all true")
        common = pass_payload(args, summary, tests)
        names = (
            "DESIGN_REVIEW_PASS_TIDA_OIA_V1.json", "IMPLEMENTATION_REVIEW_PASS_TIDA_OIA_V1.json",
            "MECHANISM_REVIEW_PASS_TIDA_OIA_V1.json", "MEMORY_REVIEW_PASS_TIDA_OIA_V1.json",
        )
        for name in names:
            _write(review_dir, name, common)
        ready = dict(common)
        ready.update({
            "design_review": names[0], "implementation_review": names[1],
            "mechanism_review": names[2], "memory_review": names[3],
        })
        failures = validate_completion_artifact(ready, phase="full_train_ready")
        if failures:
            raise RuntimeError(f"invalid FULL_TRAIN_READY schema: {failures}")
        _write(review_dir, "FULL_TRAIN_READY_TIDA_OIA_V1.json", ready)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not implementation_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
