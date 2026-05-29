from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "run_c_artifacts"
EXTERNAL = {
    "data_root": Path(r"E:\sbw\FATE_Drive\fate_oia_worktree\dataset\BDD-OIA"),
    "raw_root": Path(r"E:\sbw\FATE_Drive\fate_oia_worktree\raw_data\BDD-OIA"),
    "pretrained_weights": Path(r"E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth"),
    "grounding_cache": Path(r"E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\fate_oia_grounding_cache_20260525.jsonl"),
}
REQUIRED_ARTIFACTS = ["checkpoint_best_test.pth", "args.json", "training_config_resolved.yaml", "run_manifest.json", "metrics_best_test.json"]
REQUIRED_CODE = [
    "fate_oia/engine/train_fate_oia.py",
    "fate_oia/engine/eval_snna25.py",
    "fate_oia/models/fate_oia_model.py",
    "fate_oia/models/label_correlation.py",
    "fate_oia/datasets/bdd_oia_multitask.py",
    "utils.py",
    "vision_transformer.py",
    "configs/fate_oia_train_360x640.yaml",
]

def main() -> int:
    missing = []
    for name in REQUIRED_ARTIFACTS:
        if not (ART / name).exists():
            missing.append(str(ART / name))
    for rel in REQUIRED_CODE:
        if not (ROOT / rel).exists():
            missing.append(str(ROOT / rel))
    for name, path in EXTERNAL.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    status = {
        "root": str(ROOT),
        "artifacts_dir": str(ART),
        "external_paths": {k: str(v) for k, v in EXTERNAL.items()},
        "missing": missing,
        "ready": not missing,
    }
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if not missing else 2

if __name__ == "__main__":
    raise SystemExit(main())
