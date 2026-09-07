from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from fate_oia.explain.coev_interventions import input_intervention


def _forward(model, inputs, device: torch.device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(inputs)


def _task_metrics(y: np.ndarray, score: np.ndarray, prefix: str) -> dict[str, Any]:
    pred = score >= 0.5
    per_f1 = [f1_score(y[:, i], pred[:, i], zero_division=0) for i in range(y.shape[1])]
    per_ap = [average_precision_score(y[:, i], score[:, i]) if np.unique(y[:, i]).size > 1 else float("nan") for i in range(y.shape[1])]
    return {f"{prefix}_mF1": float(np.mean(per_f1)), f"{prefix}_oF1": float(f1_score(y.ravel(), pred.ravel(), zero_division=0)),
            f"{prefix}_mAP": float(np.nanmean(per_ap)), f"{prefix}_per_label_F1": per_f1, f"{prefix}_per_label_AP": per_ap}


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = model.training; model.eval(); rows = defaultdict(list)
    for batch_index, (inputs, targets) in enumerate(loader):
        out = _forward(model,inputs.to(device),device); rows["logits"].append(out["logits"].float().cpu())
        rows["action"].append(targets.action); rows["reason"].append(targets.reason); rows["ids"].extend(targets.ids)
        if (batch_index + 1) % 32 == 0:
            print(f'{{"event":"coev_eval","batch":{batch_index+1},"samples":{len(rows["ids"])}}}', flush=True)
    logits = torch.cat(rows["logits"]).sigmoid().numpy(); a = torch.cat(rows["action"]).numpy(); r = torch.cat(rows["reason"]).numpy()
    metrics = {**_task_metrics(a, logits[:, :4], "Act"), **_task_metrics(r, logits[:, 4:], "Exp")}
    metrics["joint"] = .5 * (metrics["Act_mF1"] + metrics["Exp_mF1"]); metrics["sample_count"] = len(a)
    metrics["ordered_id_sha256"] = hashlib.sha256("\n".join(rows["ids"]).encode()).hexdigest()
    model.train(mode)
    return metrics, {"logits": torch.from_numpy(logits), "ids": rows["ids"]}


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None, epoch: int, best_epoch: int | None) -> bool:
    if incumbent is None or candidate["joint"] > incumbent["joint"] + 1e-8: return True
    if abs(candidate["joint"] - incumbent["joint"]) <= 1e-8:
        cap = .5 * (candidate["Act_mAP"] + candidate["Exp_mAP"])
        iap = .5 * (incumbent["Act_mAP"] + incumbent["Exp_mAP"])
        return cap > iap + 1e-8 or (abs(cap - iap) <= 1e-8 and (best_epoch is None or epoch < best_epoch))
    return False


def _intervention_flips(truth: np.ndarray, full_score: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    truth = truth.astype(bool)
    full_pred = full_score >= 0.5
    pred = score >= 0.5
    return {
        "full_FP_to_intervention_TN": int((~truth & full_pred & ~pred).sum()),
        "full_TP_to_intervention_FN": int((truth & full_pred & ~pred).sum()),
        "full_TN_to_intervention_FP": int((~truth & ~full_pred & pred).sum()),
        "full_FN_to_intervention_TP": int((truth & ~full_pred & pred).sum()),
        "mean_abs_probability_delta": float(np.abs(score - full_score).mean()),
    }


@torch.no_grad()
def final_intervention_diagnostics(model, loader, device: torch.device, limit: int = 256) -> dict[str, Any]:
    views={name:{"logits":[],"a":[],"r":[]} for name in ("full","coupling_off","area_off","evidence_off","history_off","shuffle","reverse")}
    generator=torch.Generator(device=device).manual_seed(20260906)
    seen=0; mode=model.training; model.eval();all_ids=[]
    for inputs,targets in loader:
        if seen>=limit: break
        inputs=inputs.to(device); full=_forward(model,inputs,device); take=min(targets.action.shape[0],limit-seen)
        all_ids.extend(targets.ids[:take])
        generated={"full":full["logits"],"coupling_off":full["logits"]-full["factor_contribution"][...,28:].sum(-1),
                   "evidence_off":full["visual_logits"]}
        area_lift={k:v for k,v in full.items() if k in ("unary","pair","unary_valid","pair_valid")}
        area_lift["pair"]=area_lift["pair"].clone();area_lift["pair"][...,11]=0
        generated["area_off"]=model.evidence_readout(full["q_video"],area_lift)["logits"]
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            for name in ("history_off","shuffle","reverse"):
                generated[name]=input_intervention(model,inputs,name,generator=generator)["logits"]
        for name,logits in generated.items():
            views[name]["logits"].append(logits[:take].sigmoid().cpu());views[name]["a"].append(targets.action[:take]);views[name]["r"].append(targets.reason[:take])
        seen+=take
    result={}
    full_score=None
    for name,row in views.items():
        score=torch.cat(row["logits"]).numpy();a=torch.cat(row["a"]).numpy();r=torch.cat(row["r"]).numpy()
        result[name]={**_task_metrics(a,score[:,:4],"Act"),**_task_metrics(r,score[:,4:],"Exp")}
        if name=="full": full_score=score
        else:
            assert full_score is not None
            truth=np.concatenate((a,r),axis=1).astype(bool)
            result[name]["versus_full_flips"]=_intervention_flips(truth,full_score,score)
    result["audit_identity"]={"sample_count":len(all_ids),"ordered_ids":all_ids,
                              "ordered_id_sha256":hashlib.sha256("\n".join(all_ids).encode()).hexdigest(),
                              "checkpoint_view":"checkpoint_best_joint/main_logits/raw_fixed_0p5"}
    model.train(mode);return result
