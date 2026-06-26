from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ACPRTeacherLockState:
    best_joint: float = float("-inf")
    best_action: float = float("-inf")
    best_exp: float = float("-inf")
    best_epoch: int = -1


def should_accept_teacher(
    state: ACPRTeacherLockState,
    candidate_joint: float,
    candidate_action: float,
    candidate_exp: float,
    min_delta: float = 1e-4,
    action_tolerance: float = 1e-3,
    exp_tolerance: float = 1e-3,
) -> bool:
    if candidate_joint <= state.best_joint + min_delta:
        return False
    if state.best_epoch >= 0 and candidate_action < state.best_action - action_tolerance:
        return False
    if state.best_epoch >= 0 and candidate_exp < state.best_exp - exp_tolerance:
        return False
    return True


def update_teacher_if_accepted(
    threshold_head,
    state: ACPRTeacherLockState,
    epoch: int,
    candidate: dict,
    metrics: dict,
    min_delta: float = 1e-4,
    action_tolerance: float = 1e-3,
    exp_tolerance: float = 1e-3,
    ema: float = 0.20,
    copy_to_params: bool = False,
) -> dict:
    joint = float(metrics.get("joint", metrics.get("final_raw_joint", 0.0)))
    action = float(metrics.get("Act_mF1", metrics.get("action_mf1", 0.0)))
    exp = float(metrics.get("Exp_mF1", metrics.get("exp_mf1", 0.0)))
    accepted = should_accept_teacher(state, joint, action, exp, min_delta, action_tolerance, exp_tolerance)
    if accepted:
        threshold_head.update_teacher(
            candidate["threshold_logit"].to(next(threshold_head.parameters()).device),
            pred_rate_teacher=candidate["pred_rate"].to(next(threshold_head.parameters()).device),
            ema=ema,
            copy_to_params=copy_to_params,
        )
        state.best_joint = joint
        state.best_action = action
        state.best_exp = exp
        state.best_epoch = int(epoch)
    return {
        "teacher_candidate_epoch": int(epoch),
        "teacher_candidate_joint": joint,
        "teacher_candidate_action": action,
        "teacher_candidate_exp": exp,
        "teacher_accepted": bool(accepted),
        "teacher_best_epoch": int(state.best_epoch),
        "teacher_best_joint": float(state.best_joint),
        "teacher_best_action": float(state.best_action),
        "teacher_best_exp": float(state.best_exp),
    }
