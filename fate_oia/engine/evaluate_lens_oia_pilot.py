from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _faithfulness(path: Path) -> dict[str, float | bool]:
    if not path.exists():
        return {"available":False}
    data=torch.load(path,map_location="cpu",weights_only=False)
    selection=data["factor_selection"][...,:21]
    contribution=data["factor_contribution_bounded"]
    labels=data["labels_action"]
    top=selection.argmax(-1)
    selected=contribution.gather(-1,top.unsqueeze(-1)).squeeze(-1)
    wrong=contribution.gather(-1,((top+1)%21).unsqueeze(-1)).squeeze(-1)
    direction=labels*2.0-1.0
    signed=selected*direction
    state=data["factor_contribution_state"]
    state_change=(state[...,0]-state[...,1]).gather(-1,top.unsqueeze(-1)).squeeze(-1)*direction
    def lcb(value: torch.Tensor) -> float:
        flat=value.flatten().float(); return float(flat.mean()-1.645*flat.std(unbiased=False)/math.sqrt(max(1,flat.numel())))
    return {
        "available":True,
        "selected_deletion_effect":float(selected.abs().mean()),
        "equal_mass_control_effect":float(wrong.abs().mean()),
        "target_factor_effect":float(signed.mean()),
        "wrong_factor_effect":float((wrong*direction).mean()),
        "selected_direction_lcb95":lcb(signed),
        "state_swap_direction_lcb95":lcb(state_change),
    }


def evaluate(root: Path) -> dict[str, Any]:
    metrics=_jsonl(root/"metrics_summary.jsonl")
    emissions=_jsonl(root/"emission_stats.jsonl")
    flips=_jsonl(root/"synthetic_flip_audit.jsonl")
    owners=_jsonl(root/"owner_stats.jsonl")
    losses=_jsonl(root/"loss_components.jsonl")
    latest=root/"checkpoint_latest.pth"
    implementation=root.parent/"lens_oia_v1_preflight"/"LENS_IMPLEMENTATION_REVIEW.json"
    implementation_payload=json.loads(implementation.read_text(encoding="utf-8")) if implementation.exists() else {}
    gate_a=bool(latest.exists() and implementation_payload.get("pass") is True and losses and max(float(row.get("DINO_grad",1.0)) for row in losses)==0.0)
    if emissions:
        last=emissions[-1]; margin=float(last.get("ordered_margin_min",0.0))
        gate_b=margin>0.02 and float(last.get("Tplus_mean",0))>float(last.get("Tunknown_mean",0))>float(last.get("Tminus_mean",0))
    else: gate_b=False
    gate_c=bool(flips and all(float(item.get("flip_detection_AUROC",0))>=0.70 and float(item.get("gamma_unknown_flip",0))>float(item.get("gamma_unknown_clean",0)) and float(item.get("gradient_robustness_ratio",1))<=0.60 for item in flips[-1].get("rates",[])))
    non_harm=[]; improvement=[]; exp_ok=[]
    for row in metrics:
        branch=row.get("branch_metrics",{}); final=branch.get("action_final",{}); source=branch.get("action_source",{}); base=branch.get("action_base",{}); formal=branch.get("reason_formal",{}); reason_source=branch.get("reason_source",{})
        non_harm.append(float(final.get("mAP",0))>=max(float(source.get("mAP",0)),float(base.get("mAP",0)))-0.002 and float(final.get("mF1",0))>=max(float(source.get("mF1",0)),float(base.get("mF1",0)))-0.003)
        improvement.append(float(final.get("mAP",0))>=float(base.get("mAP",0))+0.001)
        exp_ok.append(float(formal.get("mAP",0))>=float(reason_source.get("mAP",0))-0.005 and float(formal.get("mF1",0))>=float(reason_source.get("mF1",0))-0.010)
    gate_d=any(all(non_harm[index:index+2]) for index in range(max(0,len(non_harm)-1))) and any(improvement)
    gate_e=bool(exp_ok and all(exp_ok[-2:]) and losses and any(float(row.get("loss_reason_latent_aux",0))>0 for row in losses))
    audit_path=root/f"epoch_{len(metrics)-1:02d}"/"audit_subset.pt" if metrics else root/"missing.pt"
    faith=_faithfulness(audit_path)
    gate_f=bool(faith.get("available") and float(faith.get("selected_deletion_effect",0))>float(faith.get("equal_mass_control_effect",0)) and float(faith.get("target_factor_effect",0))>float(faith.get("wrong_factor_effect",0)) and float(faith.get("selected_direction_lcb95",-1))>0 and float(faith.get("state_swap_direction_lcb95",-1))>0)
    required_losses={"loss_action_final","loss_action_base","loss_action_factor_aux","loss_reason_formal","loss_reason_latent_aux","loss_state","loss_emission","loss_map_anchor","loss_state_anchor","loss_view_consistency"}
    owner_delta=owners[-1].get("parameter_delta",{}) if owners else {}
    gate_g=bool(losses and required_losses.issubset(losses[-1]) and owner_delta and all(float(value)>0 for value in owner_delta.values()))
    gates={"A":gate_a,"B":gate_b,"C":gate_c,"D":gate_d,"E":gate_e,"F":gate_f,"G":gate_g}
    payload={"status":"PILOT_PASS" if all(gates.values()) else "PILOT_FAIL","gates":gates,"faithfulness":faith,"epochs_found":len(metrics),"raw_evidence":{"latest_metrics":metrics[-1] if metrics else None,"latest_emission":emissions[-1] if emissions else None,"latest_flip":flips[-1] if flips else None,"latest_owner":owners[-1] if owners else None},"missing_required":[name for name,path in {"metrics":root/"metrics_summary.jsonl","loss":root/"loss_components.jsonl","checkpoint":latest,"implementation_review":implementation}.items() if not path.exists()]}
    (root/"LENS_PILOT_GATES.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    if payload["status"]=="PILOT_PASS":
        (root/"LENS_PILOT_PASS.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",required=True); args=parser.parse_args()
    print(json.dumps(evaluate(Path(args.run_dir))))


if __name__=="__main__": main()
