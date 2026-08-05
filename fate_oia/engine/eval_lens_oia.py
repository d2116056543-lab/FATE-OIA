from __future__ import annotations

from typing import Any

import torch

from fate_oia.utils.lens_metrics import deploy_joint, multilabel_metrics


@torch.no_grad()
def evaluate_lens(model, loader, device: torch.device, *, progress: float, action_threshold: torch.Tensor | float = 0.5, reason_threshold: torch.Tensor | float = 0.5, diagnostic_samples: int = 0) -> tuple[dict[str, Any], dict[str, torch.Tensor | list[str]]]:
    model.eval(); store: dict[str, list] = {key:[] for key in ("action","reason","action_source","action_base","action_factor_aux","reason_source","reason_latent","state_prob","factor_selection","factor_contribution","unnamed_contribution","labels_action","labels_reason","file_names")}; audit={"file_names":[],"labels_action":[],"evidence_map":[],"factor_selection":[],"factor_contribution_bounded":[],"state_prob":[],"factor_contribution_state":[],"factor_local_attention":[],"branches":{}}; seen=0
    for batch in loader:
        image=batch["image"].to(device, non_blocking=True); field=model.encode_images(image); out=model.decode_from_field(field,progress=progress)
        for key, out_key in [("action","action_logits_final"),("reason","reason_logits_formal"),("action_source","action_logits_source"),("action_base","action_logits_base"),("action_factor_aux","action_logits_factor_aux"),("reason_source","reason_logits_source"),("reason_latent","reason_logits_latent"),("state_prob","state_prob"),("factor_selection","factor_selection"),("factor_contribution","factor_contribution_bounded"),("unnamed_contribution","unnamed_contribution")]: store[key].append(out[out_key].detach().cpu())
        store["labels_action"].append(batch["action"].cpu()); store["labels_reason"].append(batch["reason"].cpu()); store["file_names"].extend(batch["file_name"])
        take=min(max(0,diagnostic_samples-seen),image.shape[0])
        if take:
            audit["file_names"].extend(batch["file_name"][:take])
            audit["labels_action"].append(batch["action"][:take].cpu())
            for key in ("evidence_map","factor_selection","factor_contribution_bounded","state_prob","factor_contribution_state","factor_local_attention"):
                audit[key].append(out[key][:take].detach().cpu())
            branches=model.decode_branches_from_field(field,progress=progress,base_output=out)
            for name,value in branches.items():
                target=audit["branches"].setdefault(name,{"action":[],"reason":[]})
                target["action"].append(value["action"][:take].detach().cpu()); target["reason"].append(value["reason"][:take].detach().cpu())
            seen+=take
    merged={key:(torch.cat(value) if key != "file_names" else value) for key,value in store.items()}
    raw_action=multilabel_metrics(merged["action"],merged["labels_action"],0.5); raw_reason=multilabel_metrics(merged["reason"],merged["labels_reason"],0.5)
    deploy_action=multilabel_metrics(merged["action"],merged["labels_action"],action_threshold); deploy_reason=multilabel_metrics(merged["reason"],merged["labels_reason"],reason_threshold)
    branch_metrics={
        "action_source":multilabel_metrics(merged["action_source"],merged["labels_action"],0.5),
        "action_base":multilabel_metrics(merged["action_base"],merged["labels_action"],0.5),
        "action_final":raw_action,
        "reason_source":multilabel_metrics(merged["reason_source"],merged["labels_reason"],0.5),
        "reason_latent":multilabel_metrics(merged["reason_latent"],merged["labels_reason"],0.5),
        "reason_formal":raw_reason,
    }
    metrics={"raw_action":raw_action,"raw_reason":raw_reason,"deploy_action":deploy_action,"deploy_reason":deploy_reason,"deploy_joint":deploy_joint(deploy_action,deploy_reason),"branch_metrics":branch_metrics}
    if diagnostic_samples:
        merged_audit={key:(torch.cat(value) if key not in {"file_names","branches"} and value else value) for key,value in audit.items() if key!="branches"}
        merged_audit["branches"]={name:{kind:torch.cat(values) for kind,values in payload.items()} for name,payload in audit["branches"].items()}
        merged["audit_subset"]=merged_audit
    merged["emission_prob"]=model.annotation_emission.emission_probabilities().detach().cpu()
    return metrics, merged
