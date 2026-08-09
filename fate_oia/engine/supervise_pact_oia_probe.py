from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("PACT_SUPERVISOR", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--action-checkpoint", required=True); parser.add_argument("--reason-map-checkpoint", required=True)
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--root", default=".")
    args = parser.parse_args(); root = Path(args.root).resolve(); python = args.python
    pact_files = [str(path) for path in Path("fate_oia").rglob("*pact*.py")]
    run([python, "-m", "py_compile", *pact_files])
    run([python, "-m", "pytest", "-q", *[str(path) for path in Path("tests").glob("test_pact_*.py")]])
    run([python, "-u", "-m", "fate_oia.engine.train_pact_oia_probe",
         "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml",
         "--output-dir", ".review/pact_oia_v1_probe/counterfactual_smoke", "--source-checkpoint", args.source_checkpoint,
         "--mode", "pact", "--epochs", "1", "--batch-size", "2", "--gradient-accumulation-steps", "1",
         "--num-workers", "0", "--max-train-samples", "8", "--max-calib-samples", "4", "--max-test-samples", "4", "--device", "cuda"])
    run([python, "-m", "fate_oia.engine.audit_pact_oia_probe", "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml",
         "--source-checkpoint", args.source_checkpoint, "--output-dir", ".review/pact_oia_v1_probe", "--device", "cuda", "--full-replay"])
    run([python, "-m", "fate_oia.engine.diagnose_pact_owner_tomography",
         "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml", "--joint-checkpoint", args.source_checkpoint,
         "--action-checkpoint", args.action_checkpoint, "--reason-map-checkpoint", args.reason_map_checkpoint,
         "--output", ".review/pact_oia_v1_probe/owner_tomography.json", "--device", "cuda"])
    run([python, "-m", "fate_oia.engine.diagnose_pact_scale_conflict",
         "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml", "--source-checkpoint", args.source_checkpoint,
         "--output-dir", ".review/pact_oia_v1_probe", "--device", "cuda"])
    conflict = json.loads(Path(".review/pact_oia_v1_probe/conflict_localization.json").read_text(encoding="utf-8"))
    if not conflict["core_hypothesis_supported"]:
        raise RuntimeError("PACT core conflict hypothesis is not supported on fixed train_audit samples")
    run([python, "-m", "fate_oia.engine.profile_pact_oia",
         "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml", "--source-checkpoint", args.source_checkpoint,
         "--output", ".review/pact_oia_v1_probe/runtime_profile.json", "--device", "cuda"])
    run([python, "-m", "fate_oia.engine.train_aie_oia", "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe_control.yaml",
         "--output-dir", ".background_runs/pact_oia_v1_probe_control", "--run-kind", "pilot", "--epochs", "3",
         "--batch-size", "6", "--gradient-accumulation-steps", "5", "--num-workers", "8", "--device", "cuda",
         "--init-model-checkpoint", args.source_checkpoint])
    run([python, "-m", "fate_oia.engine.train_pact_oia_probe", "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml",
         "--output-dir", ".background_runs/pact_oia_v1_probe_method", "--source-checkpoint", args.source_checkpoint,
         "--mode", "pact", "--epochs", "3", "--batch-size", "6", "--num-workers", "8", "--device", "cuda"])
    run([python, "-m", "fate_oia.engine.evaluate_pact_oia_probe", "--control-dir", ".background_runs/pact_oia_v1_probe_control",
         "--method-dir", ".background_runs/pact_oia_v1_probe_method", "--output-dir", ".review/pact_oia_v1_probe",
         "--config", "configs/fate_oia_train_360x640_pact_oia_v1_probe.yaml", "--source-checkpoint", args.source_checkpoint])


if __name__ == "__main__":
    main()
