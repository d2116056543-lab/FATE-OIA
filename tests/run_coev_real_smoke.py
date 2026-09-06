from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import torch

from fate_oia.datasets.coev_video_dataset import CoEVVideoDataset, coev_collate
from fate_oia.engine.train_coev_oia import build_model, load_config
from fate_oia.utils.coev_preflight import source_identity


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/coev_oia_v1.yaml");parser.add_argument("--output",required=True);args=parser.parse_args()
    cfg = load_config(args.config)
    dataset = CoEVVideoDataset(cfg["data"]["manifest_path"], "test", False, max_samples=1)
    started = time.time(); row = dataset[0]; decode = time.time() - started
    inputs, _ = coev_collate([row]); model = build_model(cfg).cuda().eval()
    torch.cuda.reset_peak_memory_stats(); started = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(inputs.to("cuda"))
    torch.cuda.synchronize()
    result={"decode_seconds": decode, "forward_seconds": time.time() - started,
                      "actual_t": inputs.actual_t[0].tolist(), "valid": inputs.valid[0].tolist(),
                      "full_history_kv_length": output["full_history_kv_length"],
                      "logits_shape": list(output["logits"].shape),
                      "predicate_maps_shape": list(output["predicate_maps"].shape),
                      "factor_contribution_shape": list(output["factor_contribution"].shape),
                      "finite": bool(output["logits"].isfinite().all()),
                      "correspondence_quality_rate":float(output["correspondence_quality_rate"].mean()),
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    result["pass"]=result["finite"] and result["full_history_kv_length"]==86064 and result["logits_shape"]==[1,25]
    result["audit_identity"]=source_identity(args.config)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(result,indent=2),encoding="utf-8");temp.replace(path);print(json.dumps(result,indent=2))
    if not result["pass"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
