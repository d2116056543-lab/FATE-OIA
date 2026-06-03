from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import torch

from fate_oia.utils.psr_artifacts import first_existing, newest_matching, read_json, torch_load, write_json


@dataclass
class SpecialistCandidate:
    name: str
    role: str
    required: bool
    run_dir: Path | None
    status: str
    reason: str = ""


@dataclass
class LoadedSpecialistLogits:
    name: str
    role: str
    action_logits: torch.Tensor
    reason_logits: torch.Tensor
    labels_action: torch.Tensor
    labels_reason: torch.Tensor
    file_names: list[str]
    source_dir: Path


ACTION_FILES = [
    "logits_action_fused_best_test.pt",
    "logits_action_fused_test.pt",
    "logits/action_guarded_test.pt",
    "logits/action_final_test.pt",
    "logits/action_base_test.pt",
    "logits_action_visual_best_test.pt",
]
REASON_FILES = [
    "logits_reason_best_test.pt",
    "logits_reason_test.pt",
    "logits/reason_final_test.pt",
    "logits/reason_base_test.pt",
]
LABEL_ACTION_FILES = ["labels_action_best_test.pt", "labels_action_test.pt", "logits/labels_action_test.pt"]
LABEL_REASON_FILES = ["labels_reason_best_test.pt", "labels_reason_test.pt", "logits/labels_reason_test.pt"]
FILE_NAME_FILES = ["file_names_best_test.json", "file_names_test.json", "logits/file_names_test.json"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_epoch_dir(run_dir: Path) -> Path:
    metrics = read_json(run_dir / "metrics_best_test.json", {})
    epoch = metrics.get("epoch") if isinstance(metrics, dict) else None
    if epoch is not None:
        candidate = run_dir / f"epoch_{int(epoch):03d}"
        if candidate.exists():
            return candidate
    return run_dir


def validate_alignment(base: LoadedSpecialistLogits, other: LoadedSpecialistLogits) -> None:
    if len(base.file_names) != len(other.file_names):
        raise ValueError(f"file_names length mismatch: {base.name}={len(base.file_names)} {other.name}={len(other.file_names)}")
    if len(set(other.file_names)) != len(other.file_names):
        raise ValueError(f"duplicated file names in {other.name}")
    if base.file_names != other.file_names:
        raise ValueError(f"file_names mismatch between {base.name} and {other.name}")
    if not torch.equal(base.labels_action.cpu(), other.labels_action.cpu()):
        raise ValueError(f"action labels mismatch between {base.name} and {other.name}")
    if not torch.equal(base.labels_reason.cpu(), other.labels_reason.cpu()):
        raise ValueError(f"reason labels mismatch between {base.name} and {other.name}")


class SpecialistRegistry:
    def __init__(self, registry_config: str | Path):
        self.config_path = Path(registry_config)
        self.config = load_config(self.config_path)
        self.missing: list[dict[str, Any]] = []

    def discover(self) -> list[SpecialistCandidate]:
        candidates: list[SpecialistCandidate] = []
        cfg = self.config.get("candidates", {})
        for role in ("action_specialists", "explanation_specialists", "calibration_specialists"):
            for item in cfg.get(role, []) or []:
                globs = ((item.get("search") or {}).get("run_dir_glob") or [])
                paths = [p for p in newest_matching(globs) if p.is_dir()]
                run_dir = paths[0] if paths else None
                required = bool(item.get("required", False))
                status = "found" if run_dir else ("missing_required" if required else "missing_optional")
                reason = "" if run_dir else f"no run_dir matched {globs}"
                cand = SpecialistCandidate(item["name"], role, required, run_dir, status, reason)
                candidates.append(cand)
                if not run_dir:
                    self.missing.append({"name": cand.name, "role": role, "required": required, "reason": reason})
        return candidates

    def load_candidate(self, cand: SpecialistCandidate) -> LoadedSpecialistLogits:
        if cand.run_dir is None:
            raise FileNotFoundError(f"candidate {cand.name} has no run_dir")
        base = choose_epoch_dir(cand.run_dir)
        action_path = first_existing(base, ACTION_FILES) or first_existing(cand.run_dir, ACTION_FILES)
        reason_path = first_existing(base, REASON_FILES) or first_existing(cand.run_dir, REASON_FILES)
        la_path = first_existing(base, LABEL_ACTION_FILES) or first_existing(cand.run_dir, LABEL_ACTION_FILES)
        lr_path = first_existing(base, LABEL_REASON_FILES) or first_existing(cand.run_dir, LABEL_REASON_FILES)
        fn_path = first_existing(base, FILE_NAME_FILES) or first_existing(cand.run_dir, FILE_NAME_FILES)
        missing = [name for name, path in [("action", action_path), ("reason", reason_path), ("labels_action", la_path), ("labels_reason", lr_path), ("file_names", fn_path)] if path is None]
        if missing:
            raise FileNotFoundError(f"candidate {cand.name} missing standardized artifacts: {missing}")
        file_names = read_json(fn_path, [])
        if len(set(file_names)) != len(file_names):
            raise ValueError(f"candidate {cand.name} has duplicated file names")
        loaded = LoadedSpecialistLogits(
            name=cand.name,
            role=cand.role,
            action_logits=torch_load(action_path),
            reason_logits=torch_load(reason_path),
            labels_action=torch_load(la_path),
            labels_reason=torch_load(lr_path),
            file_names=list(file_names),
            source_dir=base,
        )
        if loaded.action_logits.ndim != 2 or loaded.action_logits.shape[1] != 4:
            raise ValueError(f"{cand.name} action logits must be [N,4], got {tuple(loaded.action_logits.shape)}")
        if loaded.reason_logits.ndim != 2 or loaded.reason_logits.shape[1] != 21:
            raise ValueError(f"{cand.name} reason logits must be [N,21], got {tuple(loaded.reason_logits.shape)}")
        return loaded

    def load_all_available(self) -> tuple[list[LoadedSpecialistLogits], list[dict[str, Any]]]:
        loaded: list[LoadedSpecialistLogits] = []
        failures: list[dict[str, Any]] = []
        for cand in self.discover():
            if cand.run_dir is None:
                failures.append({"name": cand.name, "role": cand.role, "required": cand.required, "status": cand.status, "reason": cand.reason})
                continue
            try:
                loaded.append(self.load_candidate(cand))
            except Exception as exc:
                failures.append({"name": cand.name, "role": cand.role, "required": cand.required, "status": "load_failed", "reason": str(exc)})
        return loaded, failures

    def aligned_available(self, output_dir: str | Path | None = None) -> tuple[list[LoadedSpecialistLogits], dict[str, Any]]:
        loaded, failures = self.load_all_available()
        if not any(x.role == "action_specialists" for x in loaded):
            raise RuntimeError("no action specialist available")
        if not any(x.role == "explanation_specialists" for x in loaded):
            raise RuntimeError("no explanation specialist available")
        base = loaded[0]
        for item in loaded[1:]:
            validate_alignment(base, item)
        report = {
            "base": base.name,
            "num_samples": len(base.file_names),
            "loaded": [{"name": x.name, "role": x.role, "source_dir": str(x.source_dir)} for x in loaded],
            "failures": failures,
            "aligned": True,
        }
        if output_dir:
            write_json(Path(output_dir) / "specialist_manifest.json", report)
            write_json(Path(output_dir) / "alignment_report.json", {"aligned": True, "num_samples": len(base.file_names), "base": base.name})
        return loaded, report
