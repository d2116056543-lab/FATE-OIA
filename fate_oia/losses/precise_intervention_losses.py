from __future__ import annotations

import torch
from torch.nn import functional as F


def target_specific_intervention_loss(selected_effect: torch.Tensor, control_effect: torch.Tensor, wrong_effect: torch.Tensor, base_logits: torch.Tensor, intervened_logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.10, nonreg_delta: float = 0.02, target_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    direct = (margin + control_effect - selected_effect).relu().mean()
    specific = (margin + wrong_effect - selected_effect).relu().mean()
    base = F.binary_cross_entropy_with_logits(base_logits.detach(), targets.float(), reduction="none")
    intervened = F.binary_cross_entropy_with_logits(intervened_logits, targets.float(), reduction="none")
    nonreg_raw = (intervened - base - nonreg_delta).relu()
    if target_indices is not None:
        # The selected target is expected to worsen when its evidence is
        # deleted. Non-regression therefore applies only to the other labels.
        non_target = torch.ones_like(nonreg_raw, dtype=torch.bool)
        non_target.scatter_(1, target_indices.view(-1, 1), False)
        nonreg = nonreg_raw.masked_select(non_target).mean() if non_target.any() else nonreg_raw.sum() * 0.0
    else:
        nonreg = nonreg_raw.mean()
    return {"loss_intervention_direct": direct, "loss_intervention_specific": specific, "loss_intervention_nonreg": nonreg, "loss_intervention": 0.10 * direct + 0.05 * specific + 0.05 * nonreg}


def matched_control_is_valid(selected_mask: torch.Tensor, control_mask: torch.Tensor, family_equal: torch.Tensor, sector_equal: torch.Tensor, part_equal: torch.Tensor, tolerance: float = 0.05) -> torch.Tensor:
    selected_mass = selected_mask.flatten(1).sum(1)
    control_mass = control_mask.flatten(1).sum(1)
    mass_error = (selected_mass - control_mass).abs() / selected_mass.clamp_min(1e-6)
    overlap = (selected_mask * control_mask).flatten(1).sum(1)
    return family_equal & sector_equal & part_equal & (mass_error <= tolerance) & (overlap == 0)


def incompatible_target_mask(model, task: str, field_index: torch.Tensor) -> torch.Tensor:
    """Return targets for which the selected evidence family is explicitly forbidden."""
    family_mask = model.exchange.family_mask_action if task == "action" else model.exchange.family_mask_reason
    return ~family_mask[:, field_index].transpose(0, 1)


def _empty_task_result(base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = base_logits.sum() * 0.0
    per_target = base_logits.new_zeros(base_logits.shape[-1])
    return {"loss_intervention": zero, "selected_effect_mean": zero.detach(), "control_effect_mean": zero.detach(), "wrong_effect_mean": zero.detach(), "pair_count": zero.detach(), "hard_rate": zero.detach(), "easy_rate": zero.detach(), "sign_agreement": zero.detach(), "per_target_count": per_target, "per_target_selected_sum": per_target.clone(), "per_target_control_sum": per_target.clone(), "per_target_wrong_sum": per_target.clone(), "per_target_sign_sum": per_target.clone()}


def balanced_positive_pairs(targets: torch.Tensor, max_pairs: int, deterministic: bool = False) -> torch.Tensor:
    """Sample positives round-robin by target instead of truncating row-major indices."""
    pools = []
    for target in range(targets.shape[1]):
        samples = torch.where(targets[:, target] > 0)[0]
        if samples.numel():
            if deterministic:
                samples = samples.sort().values
            else:
                samples = samples[torch.randperm(samples.numel(), device=samples.device)]
            pools.append(torch.stack([samples, torch.full_like(samples, target)], dim=1))
    selected = []
    round_index = 0
    while len(selected) < max_pairs and any(round_index < len(pool) for pool in pools):
        for pool in pools:
            if round_index < len(pool):
                selected.append(pool[round_index])
                if len(selected) == max_pairs:
                    break
        round_index += 1
    return torch.stack(selected) if selected else torch.empty(0, 2, dtype=torch.long, device=targets.device)


def _candidate_control(
    model,
    output: dict[str, torch.Tensor],
    sample_index: torch.Tensor,
    field_index: torch.Tensor,
    tolerance: float,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = output["evidence_part_coordinates"][sample_index, field_index]
    selected_mask = output["evidence_masks"][sample_index, field_index]
    part_count = model.evidence_fields.part_count[field_index]
    part_valid = torch.arange(coordinates.shape[1], device=coordinates.device).view(1, -1) < part_count.view(-1, 1)
    height, width = selected_mask.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=coordinates.device, dtype=coordinates.dtype), torch.linspace(0.0, 1.0, width, device=coordinates.device, dtype=coordinates.dtype), indexing="ij")
    selected_binary = torch.zeros_like(selected_mask, dtype=torch.bool)
    control_mask = torch.zeros_like(selected_mask)
    control_coordinates = torch.zeros_like(coordinates)
    valid = torch.ones(len(coordinates), dtype=torch.bool, device=coordinates.device)
    grid_coordinates = torch.stack([xx, yy], dim=-1).flatten(0, 1)
    for row, field in enumerate(field_index.tolist()):
        sector = model.evidence_schema[field]["sector"]
        allowed = torch.ones(height, width, dtype=torch.bool, device=coordinates.device)
        if sector == "left":
            allowed &= xx <= 0.40
        elif sector == "right":
            allowed &= xx >= 0.60
        elif sector == "center":
            allowed &= (xx >= 0.30) & (xx <= 0.70)
        elif sector == "upper":
            allowed &= yy <= 0.45
        if sector != "upper":
            allowed &= yy >= 0.25
        allowed_flat = allowed.flatten()
        allowed_index = torch.where(allowed_flat)[0]
        initial_mass = int(((selected_mask[row] > 0.5) & allowed).sum().item())
        mass = min(max(initial_mass, int(part_count[row].item())), max(1, allowed_index.numel() // 2))
        selected_rank = selected_mask[row].flatten()[allowed_index].argsort(descending=True)
        selected_support = allowed_index[selected_rank[:mass]]
        selected_flat = selected_binary[row].flatten()
        selected_flat[selected_support] = True
        available = allowed_flat & ~selected_flat
        candidate_index = torch.where(available)[0]
        if candidate_index.numel() < mass:
            valid[row] = False
            continue
        if deterministic:
            # A held-out control must not be constructed from deliberately
            # low-attention patches. Use a stateless spatial permutation so
            # repeated audits are reproducible but independent of scores.
            keys = torch.remainder(
                candidate_index.to(torch.int64) * 1103515245
                + (field + 1) * 12345
                + (row + 1) * 2654435761,
                2147483647,
            )
            support = candidate_index[keys.argsort()[:mass]]
        else:
            support = candidate_index[torch.randperm(candidate_index.numel(), device=candidate_index.device)[:mass]]
        control_mask[row].flatten()[support] = 1.0
        count = int(part_count[row].item())
        chosen = support[torch.linspace(0, len(support) - 1, count, device=support.device).round().long()]
        control_coordinates[row, :count] = grid_coordinates[chosen]
    rows = torch.arange(len(coordinates), device=coordinates.device)
    matched = matched_control_is_valid(selected_binary.float(), (control_mask > 0.5).float(), torch.ones(len(rows), dtype=torch.bool, device=rows.device), torch.ones(len(rows), dtype=torch.bool, device=rows.device), torch.ones(len(rows), dtype=torch.bool, device=rows.device), tolerance=tolerance)
    valid &= matched
    source = output["evidence_source_tokens"][sample_index].transpose(1, 2).reshape(len(rows), -1, 45, 80)
    grid = (control_coordinates.view(len(rows), -1, 1, 2) * 2.0 - 1.0).to(dtype=source.dtype)
    sampled = F.grid_sample(source, grid, mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
    control_token = (sampled * part_valid.unsqueeze(-1)).sum(1) / part_count.to(sampled).view(-1, 1)
    return control_token, control_mask, control_coordinates, valid


def _norm_matched_control_token(selected_token: torch.Tensor, control_token: torch.Tensor) -> torch.Tensor:
    """Match replacement strength to deleting the selected token.

    The planned interventions remain different (selected deletion versus a
    matched spatial replacement), but their token-space perturbation norms are
    equal so the selected-control margin cannot be won by deletion magnitude.
    """
    deletion_norm = selected_token.norm(dim=-1, keepdim=True)
    replacement_delta = control_token - selected_token
    replacement_norm = replacement_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return selected_token + replacement_delta * (deletion_norm / replacement_norm)


def _task_intervention(model, output, targets: torch.Tensor, task: str, max_pairs: int, margin: float, control_tolerance: float, deterministic: bool = False) -> dict[str, torch.Tensor]:
    attention = output[f"{task}_evidence_attention"]
    base_logits = output["action_logits_final_raw" if task == "action" else "reason_logits_semantic"]
    positive = balanced_positive_pairs(targets.float(), max_pairs, deterministic=deterministic)
    if positive.numel() == 0:
        return _empty_task_result(base_logits)
    sample_index, target_index = positive[:, 0], positive[:, 1]
    scores = attention[sample_index, target_index] * output["evidence_reliability"][sample_index]
    field_index = scores.argmax(-1)
    explicit = output["explicit_evidence_tokens"][sample_index]
    reliability = output["evidence_reliability"][sample_index]
    selected_evidence = explicit.clone()
    rows = torch.arange(len(positive), device=explicit.device)
    selected_evidence[rows, field_index] = 0.0

    control_token, _, control_coordinates, valid_control = _candidate_control(
        model, output, sample_index, field_index, control_tolerance, deterministic=deterministic
    )
    if not valid_control.any():
        return _empty_task_result(base_logits)
    sample_index = sample_index[valid_control]
    target_index = target_index[valid_control]
    field_index = field_index[valid_control]
    explicit = explicit[valid_control]
    reliability = reliability[valid_control]
    selected_evidence = selected_evidence[valid_control]
    control_token = control_token[valid_control]
    control_coordinates = control_coordinates[valid_control]
    rows = torch.arange(len(sample_index), device=explicit.device)
    control_evidence = explicit.clone()
    selected_token = explicit[rows, field_index]
    control_evidence[rows, field_index] = _norm_matched_control_token(
        selected_token, control_token.to(dtype=control_evidence.dtype)
    )

    latent_delta = output["reason_latent_delta"][sample_index]
    coordinates = output["evidence_part_coordinates"][sample_index]
    control_part_coordinates = coordinates.clone()
    control_part_coordinates[rows, field_index] = control_coordinates
    common = (
        output["action_tokens_direct"][sample_index], output["reason_tokens_direct"][sample_index],
        output["action_field_layers"][sample_index], output["reason_field_layers"][sample_index],
        output["action_logits_direct"][sample_index], output["reason_logits_direct"][sample_index],
    )
    # Selected and control use the same certificate/reliability operator. Only
    # the equal-norm token perturbation differs, preventing gate-strength bias.
    selected = model.decode_cached_intervention(*common, selected_evidence, reliability, coordinates, output["evidence_part_valid"], latent_delta)[f"{task}_logits"]
    control = model.decode_cached_intervention(*common, control_evidence, reliability, control_part_coordinates, output["evidence_part_valid"], latent_delta)[f"{task}_logits"]
    base = base_logits[sample_index]
    selected_effect = base[rows, target_index] - selected[rows, target_index]
    control_effect = base[rows, target_index] - control[rows, target_index]
    incompatible_mask = incompatible_target_mask(model, task, field_index)
    incompatible_mask[rows, target_index] = False
    has_wrong_target = incompatible_mask.any(-1)
    if not has_wrong_target.any():
        return _empty_task_result(base_logits)
    sample_index = sample_index[has_wrong_target]
    target_index = target_index[has_wrong_target]
    base = base[has_wrong_target]
    selected = selected[has_wrong_target]
    control = control[has_wrong_target]
    selected_effect = selected_effect[has_wrong_target]
    control_effect = control_effect[has_wrong_target]
    incompatible_mask = incompatible_mask[has_wrong_target]
    rows = torch.arange(len(sample_index), device=base.device)
    incompatible = base.detach().masked_fill(~incompatible_mask, -torch.inf)
    wrong_target = incompatible.argmax(-1)
    wrong_effect = base[rows, wrong_target] - selected[rows, wrong_target]
    loss = target_specific_intervention_loss(selected_effect, control_effect, wrong_effect, base, selected, targets[sample_index], margin=margin, target_indices=target_index)
    hard_rate = (selected_effect <= control_effect).float().mean()
    easy_rate = (selected_effect > control_effect + 0.10).float().mean()
    count = base.new_zeros(base.shape[-1]).scatter_add_(0, target_index, torch.ones_like(target_index, dtype=base.dtype))
    selected_sum = base.new_zeros(base.shape[-1]).scatter_add_(0, target_index, selected_effect.detach())
    control_sum = base.new_zeros(base.shape[-1]).scatter_add_(0, target_index, control_effect.detach())
    wrong_sum = base.new_zeros(base.shape[-1]).scatter_add_(0, target_index, wrong_effect.detach())
    sign = (selected_effect.detach() > 0.0).to(base.dtype)
    sign_sum = base.new_zeros(base.shape[-1]).scatter_add_(0, target_index, sign)
    return {**loss, "selected_effect_mean": selected_effect.detach().mean(), "control_effect_mean": control_effect.detach().mean(), "wrong_effect_mean": wrong_effect.detach().mean(), "pair_count": torch.tensor(float(sample_index.numel()), device=base.device), "hard_rate": hard_rate.detach(), "easy_rate": easy_rate.detach(), "sign_agreement": sign.mean(), "per_target_count": count, "per_target_selected_sum": selected_sum, "per_target_control_sum": control_sum, "per_target_wrong_sum": wrong_sum, "per_target_sign_sum": sign_sum}


def packed_target_specific_interventions(model, output, action_targets: torch.Tensor, reason_targets: torch.Tensor, max_pairs: int = 24, margin: float | None = None, control_tolerance: float | None = None, deterministic: bool = False) -> dict[str, torch.Tensor]:
    margin = float(model.intervention_margin if margin is None else margin)
    control_tolerance = float(model.intervention_control_mass_tolerance if control_tolerance is None else control_tolerance)
    action_pairs = max_pairs // 2
    action = _task_intervention(model, output, action_targets, "action", action_pairs, margin, control_tolerance, deterministic)
    reason = _task_intervention(model, output, reason_targets, "reason", max_pairs - action_pairs, margin, control_tolerance, deterministic)
    total_count = (action["pair_count"] + reason["pair_count"]).clamp_min(1.0)
    def weighted(name: str) -> torch.Tensor:
        return (action[name] * action["pair_count"] + reason[name] * reason["pair_count"]) / total_count
    return {
        "loss_intervention": action["loss_intervention"] + reason["loss_intervention"],
        "loss_intervention_action": action["loss_intervention"],
        "loss_intervention_reason": reason["loss_intervention"],
        "selected_effect_mean": weighted("selected_effect_mean"),
        "control_effect_mean": weighted("control_effect_mean"),
        "wrong_effect_mean": weighted("wrong_effect_mean"),
        "intervention_pair_count": action["pair_count"] + reason["pair_count"],
        "intervention_hard_rate": weighted("hard_rate"),
        "intervention_easy_rate": weighted("easy_rate"),
        "sign_agreement": weighted("sign_agreement"),
        **{f"action_{key}": action[key] for key in ("per_target_count", "per_target_selected_sum", "per_target_control_sum", "per_target_wrong_sum", "per_target_sign_sum")},
        **{f"reason_{key}": reason[key] for key in ("per_target_count", "per_target_selected_sum", "per_target_control_sum", "per_target_wrong_sum", "per_target_sign_sum")},
    }
