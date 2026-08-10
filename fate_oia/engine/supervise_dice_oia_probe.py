from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from fate_oia.engine.dice_common import load_config
from fate_oia.utils.dice_artifacts import validate_gate_artifacts


def run(command):
    print("DICE supervisor:"," ".join(map(str,command)),flush=True); return subprocess.run(command,check=False).returncode


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--base-checkpoint",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--epochs",type=int,default=2); p.add_argument("--device",default="cuda"); p.add_argument("--skip-oracle",action="store_true"); a=p.parse_args()
    output=Path(a.output_dir); review=output/"review"; oracle=output/"oracle"; cfg=load_config(a.config)
    checkpoint=Path(a.base_checkpoint); control_epoch=checkpoint.parent/checkpoint.stem.replace("checkpoint_","")
    replay=[sys.executable,"-m","fate_oia.engine.replay_dice_base","--config",a.config,"--base-checkpoint",a.base_checkpoint,
            "--control-epoch-dir",str(control_epoch),"--output-dir",str(output/"replay"),"--device",a.device]
    if run(replay): raise SystemExit("DICE exact base replay failed")
    shutil.copy2(output/"replay"/"DICE_BASE_REPLAY.json",output/"DICE_BASE_REPLAY.json")
    audit=[sys.executable,"-m","fate_oia.engine.audit_dice_oia_probe","--config",a.config,"--base-checkpoint",a.base_checkpoint,"--output-dir",str(review),"--device",a.device]
    if run(audit): raise SystemExit("DICE implementation audit failed")
    shutil.copy2(review/"DICE_IMPLEMENTATION_REVIEW.json",output/"DICE_IMPLEMENTATION_REVIEW.json")
    if not a.skip_oracle:
        oracle_cmd=[sys.executable,"-m","fate_oia.engine.diagnose_dice_oracle","--config",a.config,"--base-checkpoint",a.base_checkpoint,"--output-dir",str(oracle),"--device",a.device]
        if run(oracle_cmd): raise SystemExit("DICE Phase A oracle failed; Phase B is prohibited")
        shutil.copy2(oracle/"DICE_ORACLE_POTENTIAL.json",output/"DICE_ORACLE_POTENTIAL.json")
    profile=[sys.executable,"-m","fate_oia.engine.profile_dice_oia","--config",a.config,"--base-checkpoint",a.base_checkpoint,
             "--output-dir",str(output/"profile"),"--device",a.device,"--batch-size",str(cfg["data"]["batch_size"])]
    if run(profile): raise SystemExit("DICE training memory profile failed")
    train=[sys.executable,"-u","-m","fate_oia.engine.train_dice_oia_probe","--config",a.config,"--base-checkpoint",a.base_checkpoint,"--output-dir",str(output),"--epochs",str(a.epochs),"--device",a.device]
    if run(train): raise SystemExit("DICE Phase B failed")
    evaluate=[sys.executable,"-m","fate_oia.engine.evaluate_dice_oia_probe","--probe-dir",str(output),"--output-dir",str(output)]
    code=run(evaluate); missing=validate_gate_artifacts(output)
    if missing: raise SystemExit(f"DICE strict artifacts missing: {missing}")
    raise SystemExit(code)


if __name__=="__main__": main()
