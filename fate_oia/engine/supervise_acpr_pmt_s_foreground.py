from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def stream_run(cmd: list[str]) -> tuple[int, str]:
    print("[pmt_supervisor]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 200:
            tail = tail[-200:]
    return proc.wait(), "".join(tail)


def load_fallback_ladder(config_path: str, first_batch: int, first_accum: int) -> list[tuple[int, int]]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ladder = cfg.get("training", {}).get("fallback_ladder") or []
    parsed = [(int(x[0]), int(x[1])) for x in ladder if isinstance(x, list) and len(x) == 2]
    if (first_batch, first_accum) not in parsed:
        parsed.insert(0, (first_batch, first_accum))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    for p in [r"E:\sbw\FATE_Drive\task_plan.md", r"E:\sbw\FATE_Drive\findings.md", r"E:\sbw\FATE_Drive\progress.md"]:
        Path(p).read_text(encoding="utf-8", errors="ignore")
    if args.require_review_pass and not Path(".background_runs/acpr_pmt_s_v1_preflight/REVIEW_PASS_ACPR_PMT_S_V1.txt").exists():
        raise SystemExit("missing REVIEW_PASS_ACPR_PMT_S_V1.txt")
    attempts = load_fallback_ladder(args.config, args.batch_size, args.gradient_accumulation_steps)
    last_tail = ""
    for i, (batch, accum) in enumerate(attempts):
        cmd = [
            py, "-u", "-m", "fate_oia.engine.train_acpr_oia",
            "--config", args.config,
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--gradient_accumulation_steps", str(accum),
            "--device", args.device,
            "--test_only",
            "--no_feature_cache",
            "--require_no_token_compression",
        ]
        code, tail = stream_run(cmd)
        if code == 0:
            return
        last_tail = tail
        oom = "out of memory" in tail.lower() or "cuda oom" in tail.lower()
        if not oom or i == len(attempts) - 1:
            raise SystemExit(code)
        print(f"[pmt_supervisor] CUDA OOM detected; retrying fallback batch={attempts[i+1][0]} accum={attempts[i+1][1]}", flush=True)
    raise SystemExit(f"training failed after fallbacks: {last_tail[-1000:]}")


if __name__ == "__main__":
    main()
