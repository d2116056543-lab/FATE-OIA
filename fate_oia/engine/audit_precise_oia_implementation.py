from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import yaml

from fate_oia.models.precise_oia_model import PRECISEOIAModel


REQUIRED = (
    "configs/fate_oia_train_360x640_precise_oia_v1.yaml", "configs/precise_evidence_fields.yaml", "configs/precise_reason_semantics.yaml", "configs/precise_action_semantics.yaml",
    "fate_oia/datasets/bdd100k_task_aware_index.py", "fate_oia/datasets/precise_grounding_adapter.py", "fate_oia/transforms_precise.py",
    "fate_oia/models/precise_dino_field.py", "fate_oia/models/precise_visual_field.py", "fate_oia/models/precise_category_decoder.py", "fate_oia/models/precise_evidence_fields.py", "fate_oia/models/precise_visual_rereader.py", "fate_oia/models/precise_semantic_exchange.py", "fate_oia/models/precise_annotation_head.py", "fate_oia/models/precise_pcvl_probes.py", "fate_oia/models/precise_oia_model.py",
    "fate_oia/losses/precise_losses.py", "fate_oia/losses/precise_intervention_losses.py", "fate_oia/utils/precise_schema.py", "fate_oia/utils/precise_artifacts.py", "fate_oia/utils/precise_gradient_ownership.py", "fate_oia/utils/precise_runtime.py",
    "fate_oia/engine/train_precise_oia.py", "fate_oia/engine/eval_precise_oia.py", "fate_oia/engine/run_precise_pcvl.py", "fate_oia/engine/profile_precise_oia.py", "fate_oia/engine/audit_precise_oia_implementation.py", "fate_oia/engine/export_precise_cases.py", "fate_oia/engine/supervise_precise_oia_foreground.py", "scripts/FATE_OIA_precise_oia_v1_foreground.ps1",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def run_audit(config_path: str | Path, output_dir: str | Path, mode: str) -> dict:
    root = Path.cwd()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    missing = [path for path in REQUIRED if not (root / path).exists()]
    source = list((root / "fate_oia").rglob("precise_*.py"))
    compile_ok = compileall.compile_dir(root / "fate_oia", quiet=1)
    forbidden = ("ACPROIAModel", "ACPRLabelTrunk", "ACPRScenePredicateHead", "ACPRPredicateReasoner", "ACPRPairMemory", "ACPRActionComboAux", "reason_gt_to_action", "reason_logits_observed_to_action", "Start-Process", "Start-Job", "nohup", "Register-ScheduledTask")
    hits = {token: [] for token in forbidden}
    for path in source:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits[token].append(str(path.relative_to(root)))
    model = PRECISEOIAModel(Path(config_path).parent, use_mock_dino=True)
    with torch.no_grad():
        forward = model(torch.randn(1, 3, 360, 640))
    checks = {"dino_shapes": list(forward["action_logits_final_raw"].shape) == [1, 4] and list(forward["reason_logits_final_raw"].shape) == [1, 21], "dino_call_one": int(forward["diagnostics"]["dino_call_count"]) == 1, "test_only": config["eval_splits"] == "test" and config["best_selection_split"] == "test", "no_compression": config["token_compression"] == "none", "annotation_firewall_source": "semantic_logits.detach() + delta" in (root / "fate_oia/models/precise_annotation_head.py").read_text(encoding="utf-8")}
    clean_hits = {token: values for token, values in hits.items() if values}
    status = "PRE_PILOT_ELIGIBLE" if not missing and compile_ok and not clean_hits and all(checks.values()) else "CHANGES_REQUIRED"
    record = {"status": status, "git_head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"]), "base_commit": "373aa49feac17372574fd7fb056c1d79c7c848fe", "config_sha256": _sha(Path(config_path)), "skill_sha256": _sha(root / ".codex/skills/precise-oia-implementation-audit/SKILL.md"), "checked_files": list(REQUIRED), "missing": missing, "forbidden_pattern_results": clean_hits, "functional_checks": checks, "compile_ok": compile_ok, "mode": mode, "unresolved": missing + list(clean_hits)}
    (output / "implementation_audit_PRECISE_OIA_V1.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    gate = root / ".review" / "PRECISE_OIA_V1_PRE_PILOT_ELIGIBLE.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    if status == "PRE_PILOT_ELIGIBLE":
        gate.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="preflight")
    parser.add_argument("--write_pre_pilot_eligible", action="store_true")
    parser.add_argument("--write_full_train_ready", action="store_true")
    parser.add_argument("--pilot_dir")
    args = parser.parse_args()
    record = run_audit(args.config, args.output_dir, args.mode)
    print(json.dumps(record, sort_keys=True))
    if record["status"] != "PRE_PILOT_ELIGIBLE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
