from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fate_oia.metrics import multilabel_metrics_from_logits


SAVE_AUDIT_VARIANTS = (
    "utility_neutral",
    "predicate_prior_off",
    "selected_deletion",
    "matched_control",
    "evidence_only",
    "factor_identity_corruption",
    "wrong_factor_corruption",
)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _batch_names(batch: Mapping[str, Any], count: int, offset: int) -> list[str]:
    names = batch.get("file_name", batch.get("file_names"))
    if isinstance(names, (list, tuple)) and len(names) == count:
        return [str(name) for name in names]
    return [f"eval-{offset + index:08d}" for index in range(count)]


def _clone_field_with_zeroed_patches(field: Mapping[str, Any], indices: Tensor) -> dict[str, Any]:
    cloned = dict(field)
    patches = field["patch_tokens_by_layer"]
    changed = patches.clone()
    for sample in range(changed.shape[0]):
        changed[sample, :, indices[sample], :] = 0.0
    cloned["patch_tokens_by_layer"] = changed
    return cloned


def _selection_indices(output: Mapping[str, Any], *, selected: bool, count: int = 16) -> Tensor:
    mapping = output["predicate_map_action"].detach().float().amax(dim=1)
    score = mapping if selected else -mapping
    return torch.topk(score, k=min(count, score.shape[1]), dim=1).indices


def _reason_logits(output: Mapping[str, Any]) -> Tensor:
    for name in ("reason_logits_final", "reason_logits_private", "reason_logits_calalign"):
        value = output.get(name)
        if isinstance(value, Tensor):
            return value
    raise ValueError("SAVE model output has no final reason logits")


def _metrics(action_logits: Tensor, reason_logits: Tensor, action: Tensor, reason: Tensor) -> dict[str, Any]:
    action_metrics = multilabel_metrics_from_logits(action_logits, action, prefix="Act_")
    reason_metrics = multilabel_metrics_from_logits(reason_logits, reason, prefix="Exp_")
    return {**action_metrics, **reason_metrics, "joint": 0.5 * (action_metrics["Act_mF1"] + reason_metrics["Exp_mF1"])}


@torch.no_grad()
def evaluate_save_oia(
    model: nn.Module,
    test_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    fixed_audit_size: int = 128,
) -> dict[str, Any]:
    """Evaluate SAVE with branch ablations re-decoded from a single DINO field."""
    device = torch.device(device)
    model.eval()
    action_rows: list[Tensor] = []
    reason_rows: list[Tensor] = []
    base_action_rows: list[Tensor] = []
    base_reason_rows: list[Tensor] = []
    clean_reason_rows: list[Tensor] = []
    private_reason_rows: list[Tensor] = []
    evidence_delta_rows: list[Tensor] = []
    candidate_rows: list[Tensor] = []
    reliability_rows: list[Tensor] = []
    progress_zero_errors: list[float] = []
    conservation_errors: list[float] = []
    action_labels: list[Tensor] = []
    reason_labels: list[Tensor] = []
    names: list[str] = []
    branch_rows: dict[str, list[Tensor]] = {name: [] for name in SAVE_AUDIT_VARIANTS}
    audit_rows: list[dict[str, Any]] = []
    offset = 0
    before = int(getattr(model, "encode_call_count", 0))
    for source_batch in test_loader:
        batch = _move(source_batch, device)
        images, action, reason = batch["image"], batch["action"], batch["reason"]
        field = model.encode_images(images)
        main = model.decode_from_field(field, progress=1.0)
        progress_zero = model.decode_from_field(field, progress=0.0)
        progress_zero_errors.append(float((progress_zero["action_logits_final"] - progress_zero["action_logits_base"]).abs().max().cpu()))
        final_action = main["action_logits_final"]
        final_reason = _reason_logits(main)
        action_rows.append(final_action.detach().cpu())
        reason_rows.append(final_reason.detach().cpu())
        base_action_rows.append(main["action_logits_base"].detach().cpu())
        base_reason_rows.append(main["reason_logits_calalign"].detach().cpu())
        clean_reason_rows.append(main["reason_logits_clean"].detach().cpu())
        private_reason_rows.append(main["reason_logits_private_direct"].detach().cpu())
        evidence_delta_rows.append(main["action_evidence_delta"].detach().cpu())
        candidate_rows.append(main["predicate_candidate_weight_real"].detach().cpu())
        reliability_rows.append(main["reason_reliability"].detach().cpu())
        conservation_errors.append(float(main["action_conservation_error"].abs().max().cpu()))
        action_labels.append(action.detach().cpu())
        reason_labels.append(reason.detach().cpu())
        batch_names = _batch_names(source_batch, images.shape[0], offset)
        names.extend(batch_names)
        audit_count = max(0, min(images.shape[0], fixed_audit_size - len(audit_rows)))
        if audit_count:
            neutral = model.decode_from_field(field, progress=1.0, intervention={"utility_neutral": True})
            prior_off = model.decode_from_field(field, progress=1.0, intervention={"predicate_prior_off": True})
            factor_count = int(main["predicate_map_action"].shape[1])
            permutation = torch.arange(factor_count - 1, -1, -1, device=images.device)
            identity = model.decode_from_field(field, progress=1.0, intervention={"factor_permutation": permutation})
            wrong = model.decode_from_field(field, progress=1.0, intervention={"factor_permutation": permutation.roll(1)})
            selected_field = _clone_field_with_zeroed_patches(field, _selection_indices(main, selected=True))
            control_field = _clone_field_with_zeroed_patches(field, _selection_indices(main, selected=False))
            selected = model.decode_from_field(selected_field, progress=1.0)
            control = model.decode_from_field(control_field, progress=1.0)
            variants = {
                "utility_neutral": neutral["action_logits_final"],
                "predicate_prior_off": prior_off["action_logits_final"],
                "selected_deletion": selected["action_logits_final"],
                "matched_control": control["action_logits_final"],
                "evidence_only": final_action - main["action_logits_base"],
                "factor_identity_corruption": identity["action_logits_final"],
                "wrong_factor_corruption": wrong["action_logits_final"],
            }
            for name, value in variants.items():
                branch_rows[name].append(value[:audit_count].detach().cpu())
            selected_delta = (final_action[:audit_count] - selected["action_logits_final"][:audit_count]).abs()
            control_delta = (final_action[:audit_count] - control["action_logits_final"][:audit_count]).abs()
            for index in range(audit_count):
                target_action = int(torch.argmax(action[index]).item())
                target_factor = int(main["predicate_candidate_weight_real"][index, target_action].argmax().item())
                final_margin = final_action[index, target_action] - final_action[index].mean()
                evidence_margin = main["action_logits_evidence_final"][index, target_action] - main["action_logits_evidence_final"][index].mean()
                audit_rows.append({
                    "file_name": batch_names[index],
                    "selected_deletion_abs_delta": float(selected_delta[index].mean().cpu()),
                    "matched_control_abs_delta": float(control_delta[index].mean().cpu()),
                    "selected_target_delta": float(selected_delta[index, target_action].cpu()),
                    "matched_control_target_delta": float(control_delta[index, target_action].cpu()),
                    "target_action": target_action,
                    "target_factor": target_factor,
                    "final_target_margin": float(final_margin.cpu()),
                    "evidence_only_target_margin": float(evidence_margin.cpu()),
                })
        offset += images.shape[0]
    dino_calls = int(getattr(model, "encode_call_count", 0)) - before
    if dino_calls != len(action_rows):
        raise RuntimeError("SAVE evaluator encoded DINO more than once per ordinary test batch")
    final_action = torch.cat(action_rows)
    final_reason = torch.cat(reason_rows)
    base_action = torch.cat(base_action_rows)
    base_reason = torch.cat(base_reason_rows)
    clean_reason = torch.cat(clean_reason_rows)
    private_reason = torch.cat(private_reason_rows)
    action = torch.cat(action_labels)
    reason = torch.cat(reason_labels)
    branch_metrics = {
        "final": _metrics(final_action, final_reason, action, reason),
        "base": _metrics(base_action, base_reason, action, reason),
        "reason_clean": _metrics(base_action, clean_reason, action, reason),
        "reason_private_direct": _metrics(base_action, private_reason, action, reason),
    }
    for name, rows in branch_rows.items():
        if rows:
            branch_action = torch.cat(rows)
            branch_metrics[name] = _metrics(
                branch_action, final_reason[: branch_action.shape[0]],
                action[: branch_action.shape[0]], reason[: branch_action.shape[0]],
            )
    return {
        "metrics": branch_metrics["final"],
        "branch_metrics": branch_metrics,
        "logits": {
            "action_base": base_action, "action_final": final_action,
            "reason_base": base_reason, "reason_final": final_reason,
            "reason_clean": clean_reason, "reason_private_direct": private_reason,
        },
        "labels": {"action": action, "reason": reason},
        "file_names": names,
        "file_order_hash_input": "\n".join(names),
        "fixed_audit": audit_rows,
        "dino_calls": dino_calls,
        "ordinary_batches": len(action_rows),
        "mechanism": {
            "action_evidence_rms": torch.cat(evidence_delta_rows).float().square().mean(0).sqrt().tolist(),
            "candidate_max_factor_share": float(torch.cat(candidate_rows).amax(dim=-1).mean()),
            "candidate_effective_factor_count": float((torch.cat(candidate_rows).square().sum(-1).clamp_min(1e-8).reciprocal()).mean()),
            "reliability_min": float(torch.cat(reliability_rows).min()),
            "reliability_max": float(torch.cat(reliability_rows).max()),
            "progress_zero_max_abs": max(progress_zero_errors, default=float("inf")),
            "conservation_max_abs": max(conservation_errors, default=float("inf")),
        },
    }


__all__ = ["SAVE_AUDIT_VARIANTS", "evaluate_save_oia"]
