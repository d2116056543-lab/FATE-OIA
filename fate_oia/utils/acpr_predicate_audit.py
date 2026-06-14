from __future__ import annotations


def predicate_coverage_summary(predicate_targets, predicate_mask) -> dict:
    mask_sum = float(predicate_mask.sum().item()) if predicate_mask is not None else 0.0
    pos_sum = float(predicate_targets.sum().item()) if predicate_targets is not None else 0.0
    return {"predicate_mask_count": mask_sum, "predicate_positive_count": pos_sum}
