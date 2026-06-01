from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def git_head(cwd: str | Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_run_manifest(args: Any, repo_root: str | Path, train_count: int, test_count: int) -> dict[str, Any]:
    return {
        "repo": "FATE-OIA",
        "method": "SURE-OIA-v2-direct-image",
        "git_head": git_head(repo_root),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_executable": os.sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "eval_splits": ["test"],
        "best_selection_split": "test",
        "direct_image": True,
        "uses_feature_cache": False,
        "uses_val": False,
        "train_count": train_count,
        "test_count": test_count,
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "patch_size": int(args.patch_size),
        "pretrained_weights": str(args.pretrained_weights),
        "bdd100k_root": str(args.bdd100k_root),
        "data_root": str(args.data_root),
        "raw_root": str(args.raw_root),
        "losses": {
            "grounding": 0.0,
            "counterfactual": 0.0,
            "compression": "off",
            "relation_teacher_weight": float(args.relation_teacher_weight),
        },
    }
