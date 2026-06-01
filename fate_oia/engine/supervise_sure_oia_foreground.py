from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


CANONICAL_ROOT = Path("E:/sbw/FATE_Drive")
DOWNLOADS_ROOT = Path("C:/Users/WLJTXY/Downloads")


def append_progress(message: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    text = f"\n\n## {ts} SURE-OIA v2 foreground supervisor\n\n{message}\n"
    for root in [CANONICAL_ROOT, DOWNLOADS_ROOT]:
        try:
            (root / "progress.md").parent.mkdir(parents=True, exist_ok=True)
            with (root / "progress.md").open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


def stream_process(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    thread = threading.Thread(target=reader)
    thread.start()
    while True:
        item = q.get()
        if item is None:
            break
        print(item, end="", flush=True)
    return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground supervisor for SURE-OIA v2 direct-image training.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--eval_splits", default="test")
    ap.add_argument("--initial_batch_size", type=int, default=4)
    ap.add_argument("--fallback_batch_sizes", default="3,2")
    ap.add_argument("--python", default=sys.executable)
    args, extra = ap.parse_known_args()
    if args.eval_splits != "test":
        raise ValueError("SURE-OIA v2 requires eval_splits=test.")
    cwd = Path.cwd()
    out_root = Path(args.output_dir)
    batch_candidates = [args.initial_batch_size] + [int(x) for x in args.fallback_batch_sizes.split(",") if x.strip()]
    append_progress(f"启动前台监督。output_dir={out_root}; batch candidates={batch_candidates}; eval_splits=test")
    for batch in batch_candidates:
        accum = max(1, round(32 / batch))
        run_dir = out_root if batch == args.initial_batch_size else out_root.with_name(out_root.name + f"_fallback_b{batch}")
        cmd = [
            args.python,
            "-m",
            "fate_oia.engine.train_sure_oia",
            "--config",
            args.config,
            "--output_dir",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--eval_splits",
            "test",
            "--batch_size",
            str(batch),
            "--gradient_accumulation_steps",
            str(accum),
        ] + extra
        append_progress(f"前台启动训练命令：batch={batch}, accumulation={accum}, dir={run_dir}")
        code = stream_process(cmd, cwd)
        if code == 0:
            append_progress(f"SURE-OIA v2 训练进程正常结束。batch={batch}, output={run_dir}")
            return
        append_progress(f"SURE-OIA v2 训练进程返回非零 code={code}。准备尝试下一个 batch fallback。")
    raise SystemExit("All SURE-OIA v2 batch candidates failed.")


if __name__ == "__main__":
    main()
