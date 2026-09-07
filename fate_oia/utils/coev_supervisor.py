from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


def atomic_status(path: Path, payload: dict) -> None:
    temp=path.with_suffix(path.suffix+".tmp");temp.parent.mkdir(parents=True,exist_ok=True)
    temp.write_text(json.dumps(payload,indent=2),encoding="utf-8");temp.replace(path)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--resume");a=p.parse_args()
    ready=Path(".review/coev_v1/FULL_TRAIN_READY.json")
    payload=json.loads(ready.read_text(encoding="utf-8")) if ready.is_file() else {}
    if not payload.get("pass"):
        raise RuntimeError("bound FULL_TRAIN_READY is required")
    launch=payload["launch_configuration"]
    command=[sys.executable,"-u","-m","fate_oia.engine.train_coev_oia","--config",a.config,"--ready-manifest",str(ready),
             "--batch-size",str(launch["batch_size"]),"--grad-accum",str(launch["gradient_accumulation_steps"])]
    if a.resume: command += ["--resume",a.resume]
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"));output=Path(cfg["runtime"]["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    free_gib=__import__("shutil").disk_usage(output).free/2**30
    if free_gib<20: raise RuntimeError(f"insufficient output disk space: {free_gib:.2f} GiB")
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    status_path=output/"supervisor_status.json";run_id=f"coev-{int(time.time())}-{process.pid}"
    lines: queue.Queue[str]=queue.Queue();console_lines: queue.Queue[str]=queue.Queue(maxsize=256);assert process.stdout is not None
    def consume() -> None:
        for line in process.stdout: lines.put(line)
    def consume_console() -> None:
        while True:
            line=console_lines.get()
            try: print(line,end="",flush=True)
            except (BrokenPipeError,OSError): return
    reader=threading.Thread(target=consume,name="coev-stdout-reader");reader.start();last=time.time()
    threading.Thread(target=consume_console,name="coev-console-writer",daemon=True).start()
    atomic_status(status_path,{"run_id":run_id,"parent_pid":os.getpid(),"child_pid":process.pid,"attached":True,"command":command,"started":last,"last_event":last})
    log_path=output/"full_train.log"
    try:
        with log_path.open("a",encoding="utf-8",buffering=1) as log:
            while process.poll() is None or not lines.empty():
                try:
                    line=lines.get(timeout=60);log.write(line);log.flush();last=time.time()
                    atomic_status(status_path,{"run_id":run_id,"parent_pid":os.getpid(),"child_pid":process.pid,"attached":True,"command":command,"last_event":last})
                    try: console_lines.put_nowait(line)
                    except queue.Full: pass
                except queue.Empty:
                    heartbeat=json.dumps({"event":"coev_supervisor_heartbeat","run_id":run_id,"child_pid":process.pid,"seconds_since_output":time.time()-last})+"\n"
                    log.write(heartbeat);log.flush()
                    atomic_status(status_path,{"run_id":run_id,"parent_pid":os.getpid(),"child_pid":process.pid,"attached":True,"command":command,"last_event":time.time(),"seconds_since_child_output":time.time()-last})
                    try: console_lines.put_nowait(heartbeat)
                    except queue.Full: pass
        code=process.wait();reader.join()
    except KeyboardInterrupt:
        process.terminate();code=process.wait();reader.join()
        atomic_status(status_path,{"run_id":run_id,"parent_pid":os.getpid(),"child_pid":process.pid,"attached":False,"interrupted":True,"returncode":code,"last_event":time.time()})
        raise
    atomic_status(status_path,{"run_id":run_id,"parent_pid":os.getpid(),"child_pid":process.pid,"attached":False,"returncode":code,"last_event":time.time()})
    if code: raise SystemExit(code)


if __name__=="__main__": main()
