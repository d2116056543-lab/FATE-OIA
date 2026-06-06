from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def git_head(cwd: str | Path = ".") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return "unknown"


def write_run_manifest(path: str | Path, args: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo": "FATE-OIA",
        "method": "DIVA-CAF-OIA V2",
        "git_head": git_head(Path.cwd()),
        "python": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "direct_image": True,
        "feature_cache": False,
        "eval_protocol": "test-only",
        "best_selection_split": "test",
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }
    if extra:
        manifest.update(extra)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
