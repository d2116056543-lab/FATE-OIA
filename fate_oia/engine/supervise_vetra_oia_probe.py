from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(args):
    print("VETRA_SUPERVISOR:", " ".join(map(str,args)), flush=True)
    subprocess.run(args, check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--epochs",type=int,default=3); p.add_argument("--device",default="cuda")
    p.add_argument("--screening",action="store_true"); a=p.parse_args(); root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
    common=["--config",a.config,"--source-checkpoint",a.source_checkpoint,"--output-dir",str(root),"--device",a.device]
    _run([sys.executable,"-m","fate_oia.engine.audit_vetra_oia_probe",*common])
    _run([sys.executable,"-m","fate_oia.engine.replay_vetra_source",*common])
    _run([sys.executable,"-m","fate_oia.engine.profile_vetra_oia",*common])
    train=[sys.executable,"-u","-m","fate_oia.engine.train_vetra_oia_probe",*common,"--epochs",str(a.epochs),"--batch-size","6","--gradient-accumulation-steps","5","--num-workers","8"]
    if a.screening:
        train += ["--max-train-main-samples","512","--max-calib-samples","256","--max-audit-samples","256","--max-test-samples","512"]
    _run(train)


if __name__ == "__main__": main()
