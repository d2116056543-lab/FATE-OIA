from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",required=True); args=parser.parse_args(); root=Path(args.run_dir)
    metrics=root/"metrics_summary.jsonl"; loss=root/"loss_components.jsonl"; latest=root/"checkpoint_latest.pth"
    gates={"A":latest.exists(),"B":loss.exists(),"C":loss.exists(),"D":metrics.exists(),"E":metrics.exists(),"F":False,"G":False}
    # F/G require fixed audit subset deletion and explicit owner artifacts; no self-declared pass.
    payload={"status":"PILOT_PASS" if all(gates.values()) else "PILOT_FAIL","gates":gates,"missing_required":[name for name,path in {"metrics":metrics,"loss":loss,"checkpoint":latest}.items() if not path.exists()]}
    (root/"LENS_PILOT_GATES.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload))


if __name__=="__main__": main()
