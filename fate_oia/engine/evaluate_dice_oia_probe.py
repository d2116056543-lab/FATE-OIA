from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.dice_artifacts import write_json
from fate_oia.utils.pact_bootstrap import paired_bootstrap


def _metric(rows):
    return aie_branch_metrics(torch.from_numpy(rows["action"]),torch.from_numpy(rows["reason"]),torch.from_numpy(rows["action_target"]),torch.from_numpy(rows["reason_target"]))


SCALAR_METRICS = ("Act_mF1", "Act_oF1", "Act_mAP", "Exp_mF1", "Exp_oF1", "Exp_mAP", "joint")


def _scalar_metric(rows):
    metrics = _metric(rows)
    return {name: metrics[name] for name in SCALAR_METRICS}


def _thresholded_rows(rows, threshold):
    threshold = torch.as_tensor(threshold)
    return {
        **rows,
        "action": apply_posthoc_threshold(torch.from_numpy(rows["action"]), threshold[:4]).numpy(),
        "reason": apply_posthoc_threshold(torch.from_numpy(rows["reason"]), threshold[4:]).numpy(),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--probe-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--control-epoch-dir"); p.add_argument("--resamples",type=int,default=2000); p.add_argument("--seed",type=int,default=20260810); a=p.parse_args()
    source_threshold = None
    if a.control_epoch_dir:
        source_branch = json.loads((Path(a.control_epoch_dir)/"branch_metrics.json").read_text(encoding="utf-8"))
        source_threshold = source_branch["thresholds_train_calib"] if "thresholds_train_calib" in source_branch else source_branch["calibration_thresholds"]["threshold_prob"]
    root=Path(a.probe_dir); decisions=[]
    for epoch_dir in sorted(root.glob("epoch_*")):
        base={"action":torch.load(epoch_dir/"action_logits_base_test.pt",map_location="cpu").numpy(),"reason":torch.load(epoch_dir/"reason_logits_base_test.pt",map_location="cpu").numpy(),"action_target":torch.load(epoch_dir/"labels_action_test.pt",map_location="cpu").numpy(),"reason_target":torch.load(epoch_dir/"labels_reason_test.pt",map_location="cpu").numpy()}
        dice={**base,"action":torch.load(epoch_dir/"action_logits_dice_test.pt",map_location="cpu").numpy(),"reason":torch.load(epoch_dir/"reason_logits_dice_test.pt",map_location="cpu").numpy()}
        branch=json.loads((epoch_dir/"branch_metrics.json").read_text(encoding="utf-8"))
        base_deploy=_thresholded_rows(base,branch["base_thresholds_train_calib"])
        dice_deploy=_thresholded_rows(dice,branch["dice_thresholds_train_calib"])
        source_base=_thresholded_rows(base,source_threshold) if source_threshold is not None else None
        source_dice=_thresholded_rows(dice,source_threshold) if source_threshold is not None else None
        boot=paired_bootstrap(base_deploy,dice_deploy,_scalar_metric,a.resamples,a.seed+len(decisions))
        decisions.append({"epoch":len(decisions),"base_raw":_metric(base),"dice_raw":_metric(dice),
                          "base_deploy":_metric(base_deploy),"dice_deploy":_metric(dice_deploy),
                          "base_source_fixed":_metric(source_base) if source_base is not None else None,
                          "dice_source_fixed":_metric(source_dice) if source_dice is not None else None,
                          "bootstrap":boot})
    if not decisions: raise RuntimeError("no DICE epoch artifacts found")
    best=max(decisions,key=lambda x:(x["dice_source_fixed"] or x["dice_deploy"])["Act_mF1"])
    point_base=best["base_source_fixed"] or best["base_deploy"]
    point_dice=best["dice_source_fixed"] or best["dice_deploy"]
    point=(point_dice["Act_mF1"]>=.729 and point_dice["Act_mF1"]>=point_base["Act_mF1"]+.002 and point_dice["Act_mAP"]>=point_base["Act_mAP"]-.0005 and point_dice["Act_oF1"]>=point_base["Act_oF1"]-.001 and point_dice["joint"]>=point_base["joint"]+.001)
    rank=(best["dice_raw"]["Act_mF1"]>=best["base_raw"]["Act_mF1"]+.001 and best["dice_raw"]["Act_mAP"]>=best["base_raw"]["Act_mAP"]-.0005 and sum(d>=b for d,b in zip(best["dice_raw"]["Act_per_label_ap"],best["base_raw"]["Act_per_label_ap"]))>=3)
    bootstrap=best["bootstrap"]["Act_mF1"]["mean"]>0 and best["bootstrap"]["Act_mAP"]["p2_5"]>=-.001 and best["bootstrap"]["joint"]["mean"]>0
    mechanism_path=root/"DICE_MECHANISM_GATES.json"
    mechanism=json.loads(mechanism_path.read_text(encoding="utf-8")) if mechanism_path.exists() else {}
    action_events=mechanism.get("per_action",{})
    mechanism_gate=(mechanism.get("valid_events",0)>=1000 and len(action_events)==4
        and all(action_events.get(str(action),{}).get("count",0)>0 for action in range(4))
        and (mechanism.get("certificate_mean") or 0)>0
        and (mechanism.get("certificate_positive_rate_lcb95") or 0)>.55
        and (mechanism.get("license_prediction_auc") or 0)>=.65
        and (mechanism.get("contribution_effect_spearman") or -1)>=.30)
    reason_identity=all(torch.equal(torch.load(epoch_dir/"reason_logits_base_test.pt",map_location="cpu"),
                                    torch.load(epoch_dir/"reason_logits_dice_test.pt",map_location="cpu"))
                        for epoch_dir in sorted(root.glob("epoch_*")))
    result={"pass":point and rank and bootstrap and mechanism_gate and reason_identity,
            "point_metrics":point,"raw_ranking":rank,"paired_bootstrap":bootstrap,
            "mechanism_gate":mechanism_gate,"reason_exact_identity":reason_identity,
            "mechanism":mechanism,"epochs":decisions,"best_epoch":best["epoch"]}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); write_json(out/"DICE_PROBE_METRICS.json",result); write_json(out/"DICE_PAIRED_BOOTSTRAP.json",best["bootstrap"])
    write_json(out/("DICE_FAST_VALIDATION_PASS.json" if result["pass"] else "DICE_FAST_VALIDATION_FAIL.json"),result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
