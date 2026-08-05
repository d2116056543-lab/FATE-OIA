from __future__ import annotations

import torch


def _binary_ranking_metrics(score: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    target=target.bool(); positives=int(target.sum()); negatives=int((~target).sum())
    if positives == 0:
        return 0.0, 0.0
    order=torch.argsort(score,descending=True); ranked=target[order].float()
    precision=ranked.cumsum(0)/torch.arange(1,ranked.numel()+1,device=score.device)
    ap=float((precision*ranked).sum()/positives)
    if negatives == 0:
        return ap, 0.0
    ascending=torch.argsort(score); ranks=torch.empty_like(ascending,dtype=torch.float32); ranks[ascending]=torch.arange(1,score.numel()+1,device=score.device,dtype=torch.float32)
    auc=float((ranks[target].sum()-positives*(positives+1)/2)/(positives*negatives))
    return ap,auc


def multilabel_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: torch.Tensor | float = 0.5) -> dict[str, float | list[float]]:
    probs = logits.sigmoid(); prediction = probs >= threshold
    target = labels.bool(); tp = (prediction & target).sum(0).float(); fp = (prediction & ~target).sum(0).float(); fn = (~prediction & target).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
    precision = tp / (tp + fp).clamp_min(1.0); recall = tp / (tp + fn).clamp_min(1.0)
    overall = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1.0)
    ranking=[_binary_ranking_metrics(probs[:,index],target[:,index]) for index in range(probs.shape[1])]
    per_ap=[item[0] for item in ranking]; per_auc=[item[1] for item in ranking]
    return {"mF1": float(f1.mean()), "oF1": float(overall), "mAP":sum(per_ap)/len(per_ap),"mAUC":sum(per_auc)/len(per_auc),"per_label_F1": f1.tolist(),"per_label_AP":per_ap,"per_label_AUC":per_auc,"precision": float(precision.mean()), "recall": float(recall.mean())}


def deploy_joint(action: dict[str, float], reason: dict[str, float]) -> float:
    return 0.5 * float(action["mF1"]) + 0.5 * float(reason["mF1"])
