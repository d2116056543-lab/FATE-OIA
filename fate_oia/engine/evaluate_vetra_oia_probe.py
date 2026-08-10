from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.vetra_artifacts import write_json


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _macro_f1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits > 0
    tp=(pred & target.bool()).sum(0).float(); fp=(pred & ~target.bool()).sum(0).float(); fn=(~pred & target.bool()).sum(0).float()
    return float((2*tp/(2*tp+fp+fn).clamp_min(1)).mean())


def _macro_ap(logits: torch.Tensor, target: torch.Tensor) -> float:
    y=target.numpy(); score=logits.numpy()
    return float(np.mean([average_precision_score(y[:,i],score[:,i]) for i in range(y.shape[1])]))


def paired_bootstrap(base_action, final_action, reason, action_target, reason_target, threshold, resamples=2000, seed=20260810):
    base_deploy=apply_posthoc_threshold(base_action,threshold[:4]); final_deploy=apply_posthoc_threshold(final_action,threshold[:4])
    reason_deploy=apply_posthoc_threshold(reason,threshold[4:]); n=len(action_target); rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(resamples):
        idx=torch.from_numpy(rng.integers(0,n,n))
        da=_macro_f1(final_deploy[idx],action_target[idx])-_macro_f1(base_deploy[idx],action_target[idx])
        dr=0.0  # reason is bit-exact, retained explicitly for joint accounting.
        dm=_macro_ap(final_action[idx],action_target[idx])-_macro_ap(base_action[idx],action_target[idx])
        deltas.append((da,.5*(da+dr),dm))
    values=np.asarray(deltas)
    return {"resamples":resamples,"act_mf1_delta_ci95":np.quantile(values[:,0],[.025,.975]).tolist(),
            "joint_delta_ci95":np.quantile(values[:,1],[.025,.975]).tolist(),
            "act_map_delta_ci95":np.quantile(values[:,2],[.025,.975]).tolist()}


def finalize_probe(root: Path, cfg: dict) -> dict:
    epochs=sorted(root.glob("epoch_*"));
    if not epochs: raise RuntimeError("no VETRA epoch artifacts")
    source=_load_json(Path(cfg["experiment"]["source_control_dir"])/"branch_metrics.json")["deploy"]
    rows=[]
    for epoch in epochs:
        metrics=_load_json(epoch/"branch_metrics.json"); current=metrics["vetra_source_fixed"]
        rows.append({"epoch":int(epoch.name.split("_")[-1]),"metrics":current,"raw":metrics["vetra_raw"],"own":metrics["vetra_deploy"]})
    best=max(rows,key=lambda row: row["metrics"]["Act_mF1"]); epoch=root/f"epoch_{best['epoch']:03d}"
    base=torch.load(epoch/"action_logits_base_test.pt",weights_only=True); final=torch.load(epoch/"action_logits_vetra_test.pt",weights_only=True)
    reason=torch.load(epoch/"reason_logits_vetra_test.pt",weights_only=True); ay=torch.load(epoch/"labels_action_test.pt",weights_only=True); ry=torch.load(epoch/"labels_reason_test.pt",weights_only=True)
    threshold=torch.tensor(_load_json(epoch/"branch_metrics.json")["source_fixed_thresholds"])
    bootstrap=paired_bootstrap(base,final,reason,ay,ry,threshold,int(cfg["bootstrap"]["resamples"]),int(cfg["bootstrap"]["seed"]))
    ablation=_load_json(epoch/"ablation_metrics.json"); cf=_load_json(epoch/"counterfactual_audit.json")
    route_rows=_load_json(epoch/"route_stats.json"); route={k:float(np.mean([r[k] for r in route_rows])) for k in route_rows[0]} if route_rows else {}
    current=best["metrics"]; raw=best["raw"]
    per_ap=np.asarray(raw["Act_per_label_ap"])-np.asarray(source["Act_per_label_ap"])
    reason_exact=abs(current["Exp_mF1"]-source["Exp_mF1"])<1e-9 and abs(current["Exp_mAP"]-source["Exp_mAP"])<1e-9
    mechanism={"best_epoch":best["epoch"],"source":source,"best":current,"raw":raw,
               "per_action_ap_delta":per_ap.tolist(),"reason_exact":reason_exact,"route_mean":route,
               "ablation":ablation,"counterfactual":cf}
    write_json(root/"VETRA_ABLATION_METRICS.json",ablation); write_json(root/"VETRA_COUNTERFACTUAL_AUDIT.json",cf)
    write_json(root/"VETRA_PAIRED_BOOTSTRAP.json",bootstrap); write_json(root/"VETRA_MECHANISM_SCREEN.json",mechanism)
    go=(current["Act_mF1"]>=.729 and raw["Act_mAP"]>=source["Act_mAP"]-.0003
        and current["Act_oF1"]>=source["Act_oF1"]-.0005 and reason_exact
        and int((per_ap>=-.0015).sum())==4 and bootstrap["act_map_delta_ci95"][0]>=-.0005)
    decision={"pass":bool(go),"best_epoch":best["epoch"],"best_source_fixed":current,"source":source,
              "bootstrap":bootstrap,"reason":"metrics_and_mechanism_pass" if go else "fast_probe_did_not_meet_safe_improvement"}
    pass_path=root/"VETRA_FAST_VALIDATION_PASS.json"; fail_path=root/"VETRA_FAST_VALIDATION_FAIL.json"
    if go:
        fail_path.unlink(missing_ok=True); write_json(pass_path,decision)
    else:
        pass_path.unlink(missing_ok=True); write_json(fail_path,decision)
    return decision


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")); print(json.dumps(finalize_probe(Path(a.output_dir),cfg),indent=2))


if __name__ == "__main__": main()
