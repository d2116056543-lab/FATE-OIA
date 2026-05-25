from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_kl_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """KL distillation for multi-label logits using a two-class Bernoulli view."""
    t = float(temperature)
    s_prob = torch.stack([1.0 - torch.sigmoid(student_logits / t), torch.sigmoid(student_logits / t)], dim=-1).clamp(1e-6, 1.0)
    q_prob = torch.stack([1.0 - torch.sigmoid(teacher_logits / t), torch.sigmoid(teacher_logits / t)], dim=-1).clamp(1e-6, 1.0)
    return F.kl_div(s_prob.log(), q_prob, reduction="batchmean") * (t * t)


def attention_map_distillation(student_attention: torch.Tensor, teacher_attention: torch.Tensor) -> torch.Tensor:
    if student_attention.shape != teacher_attention.shape:
        raise ValueError(f"attention shape mismatch: {tuple(student_attention.shape)} vs {tuple(teacher_attention.shape)}")
    return F.mse_loss(student_attention.float(), teacher_attention.float())


def fate_oia_distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    *,
    lambda_logits: float = 1.0,
    lambda_attention: float = 0.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Clean interface for future full-token teacher -> compressed student training."""
    loss = None
    for key in ("action_visual_logits", "action_reason_logits", "action_fused_logits", "reason_logits"):
        if key in student and key in teacher:
            item = logits_kl_distillation(student[key], teacher[key], temperature=temperature)
            loss = item if loss is None else loss + item
    if loss is None:
        first = next(iter(student.values()))
        loss = first.new_zeros(())
    loss = float(lambda_logits) * loss
    if lambda_attention > 0 and "attention" in student and "attention" in teacher:
        loss = loss + float(lambda_attention) * attention_map_distillation(student["attention"], teacher["attention"])
    return loss
