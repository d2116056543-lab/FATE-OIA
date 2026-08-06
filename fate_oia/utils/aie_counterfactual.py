from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def target_signed_margin(logits: Tensor, targets: Tensor) -> Tensor:
    return (2.0 * targets - 1.0) * logits


def straight_through_topk_mask(probability: Tensor, topk: int = 64) -> Tensor:
    k = min(int(topk), probability.shape[-1])
    index = probability.topk(k, dim=-1).indices
    hard = torch.zeros_like(probability).scatter_(-1, index, 1.0)
    soft = probability * (float(k) / probability.sum(-1, keepdim=True).clamp_min(1e-8))
    soft = soft.clamp(0, 1)
    return hard.detach() - soft.detach() + soft


def deterministic_seed(file_name: str, action_id: int, probe_id: int, global_update: int) -> int:
    payload = f"{file_name}|{action_id}|{probe_id}|{global_update}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def matched_control_mask(selected: Tensor, region_mask: Tensor, seed: int, max_overlap: float = 0.20) -> tuple[Tensor, bool, float]:
    support = int(selected.sum().item())
    candidates = torch.where(region_mask > 0.5)[0]
    if support <= 0 or candidates.numel() < support:
        return torch.zeros_like(selected), False, 1.0
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best, best_overlap = None, 1.0
    for _ in range(4):
        perm = candidates[torch.randperm(candidates.numel(), generator=generator, device="cpu").to(candidates.device)[:support]]
        control = torch.zeros_like(selected).scatter_(0, perm, 1.0)
        overlap = float((control * selected).sum().item() / max(support, 1))
        if overlap < best_overlap:
            best, best_overlap = control, overlap
    valid = best is not None and best_overlap <= float(max_overlap)
    return (best if best is not None else torch.zeros_like(selected)), valid, best_overlap


@dataclass
class AIECounterfactualConfig:
    batch_fraction: float = 0.50
    max_actions_per_sample: int = 2
    max_atoms_per_event: int = 8
    topk_patches: int = 64
    max_control_overlap: float = 0.20
    necessity_margin: float = 0.05
    sufficiency_ratio: float = 0.50
    sufficiency_margin: float = 0.05


class AIECounterfactualEngine:
    """Same-field intervention engine. It never calls image encoding or primary decoding."""

    def __init__(self, config: AIECounterfactualConfig | None = None) -> None:
        self.config = config or AIECounterfactualConfig()

    @staticmethod
    def _slice_fixed(output: dict[str, Any], index: int) -> dict[str, Tensor]:
        keys = ("action_nodes_primary", "action_logits_primary", "predicate_attention", "predicate_probs")
        return {key: output[key][index : index + 1] for key in keys}

    @staticmethod
    def _region_name(reference: Tensor) -> str:
        x, y = (float(value) for value in reference.detach().cpu())
        if y < 0.45:
            return "upper_traffic_region"
        if x < 0.42 and y > 0.35:
            return "left_corridor"
        if x > 0.58 and y > 0.35:
            return "right_corridor"
        if y > 0.65:
            return "bottom_drivable_region"
        return "front_center"

    @staticmethod
    def _substitute(field: Tensor, mask: Tensor, region_mask: Tensor, keep_only: bool = False) -> Tensor:
        region = region_mask.to(field.device, field.dtype).view(1, 1, -1, 1)
        background = (field * region).sum(2, keepdim=True) / region.sum(2, keepdim=True).clamp_min(1e-8)
        patch_mask = mask.view(1, 1, -1, 1)
        return background + patch_mask * (field - background) if keep_only else field - patch_mask * (field - background)

    def run(
        self,
        model,
        output: dict[str, Any],
        action_target: Tensor,
        file_names: list[str],
        *,
        global_update: int,
        action_scale: float,
    ) -> dict[str, Any]:
        batch = action_target.shape[0]
        selected_batch = min(batch, max(1, int(batch * self.config.batch_fraction)))
        signed_contribution = (2 * action_target - 1)[..., None] * output["bounded_contribution"]
        rows: list[dict[str, Any]] = []
        selected_drops, control_drops, support_values, valid_values, sufficiency_values = [], [], [], [], []
        selected_masks, control_masks, selected_logits, control_logits = [], [], [], []
        keep_masks, keep_logits, wrong_probe_masks, wrong_action_masks = [], [], [], []
        wrong_probe_logits, wrong_action_logits = [], []
        wrong_probe_drops, wrong_action_drops = [], []
        invalid: dict[str, int] = {}
        atom_count = 0
        for sample in range(selected_batch):
            signed_margin = (2 * action_target[sample] - 1) * output["action_logits_final_train"][sample]
            current_loss = torch.nn.functional.softplus(-signed_margin.float())
            support = signed_contribution[sample].max(-1).values.float()
            score = current_loss + 0.25 * (support - support.mean()) / support.std().clamp_min(1e-6)
            ranked = torch.argsort(score, descending=True)
            forced_action = int((global_update + sample) % action_target.shape[1])
            action_order = torch.cat((ranked.new_tensor([forced_action]), ranked[ranked != forced_action]))
            for action_id_tensor in action_order[: self.config.max_actions_per_sample]:
                if atom_count >= self.config.max_atoms_per_event:
                    break
                action_id = int(action_id_tensor)
                probe_id = int(signed_contribution[sample, action_id].argmax())
                probability = output["evidence_map"][sample, action_id, probe_id]
                region_name = self._region_name(output["reference_point"][sample, action_id, probe_id])
                region = output["ego_region_masks"][region_name]
                selected = straight_through_topk_mask(probability * region.to(probability), self.config.topk_patches)
                selected_hard = (selected.detach() > 0.5).to(selected.dtype)
                seed = deterministic_seed(file_names[sample], action_id, probe_id, global_update)
                control, valid, overlap = matched_control_mask(selected_hard, region, seed, self.config.max_control_overlap)
                if not valid:
                    invalid["control_overlap_or_support"] = invalid.get("control_overlap_or_support", 0) + 1
                    continue
                base_field = output["patch_tokens_by_layer_raw"][sample : sample + 1].detach()
                fixed = self._slice_fixed(output, sample)
                selected_field = self._substitute(base_field, selected, region)
                control_field = self._substitute(base_field, control, region)
                selected_out = model.rerun_action_evidence_from_field(
                    {"patch_tokens_by_layer_raw": selected_field}, fixed,
                    action_scale=action_scale, predicate_bias_enabled=True,
                )
                control_out = model.rerun_action_evidence_from_field(
                    {"patch_tokens_by_layer_raw": control_field}, fixed,
                    action_scale=action_scale, predicate_bias_enabled=True,
                )
                sign = 2 * action_target[sample, action_id] - 1
                original_margin = sign * output["action_logits_final_train"][sample, action_id]
                selected_margin = sign * selected_out["action_logits_final_train"][0, action_id]
                control_margin = sign * control_out["action_logits_final_train"][0, action_id]
                selected_drop = original_margin.float() - selected_margin.float()
                control_drop = original_margin.float() - control_margin.float()
                wrong_probe = (probe_id + 1) % output["evidence_map"].shape[2]
                wrong_action = (action_id + 1) % output["evidence_map"].shape[1]
                wrong_probe_mask = straight_through_topk_mask(
                    output["evidence_map"][sample, action_id, wrong_probe] * region.to(probability), self.config.topk_patches
                )
                wrong_action_mask = straight_through_topk_mask(
                    output["evidence_map"][sample, wrong_action, probe_id] * region.to(probability), self.config.topk_patches
                )
                wrong_probe_out = model.rerun_action_evidence_from_field(
                    {"patch_tokens_by_layer_raw": self._substitute(base_field, wrong_probe_mask, region)}, fixed,
                    action_scale=action_scale, predicate_bias_enabled=True,
                )
                wrong_action_out = model.rerun_action_evidence_from_field(
                    {"patch_tokens_by_layer_raw": self._substitute(base_field, wrong_action_mask, region)}, fixed,
                    action_scale=action_scale, predicate_bias_enabled=True,
                )
                wrong_probe_drop = original_margin.float() - sign * wrong_probe_out["action_logits_final_train"][0, action_id].float()
                wrong_action_drop = original_margin.float() - sign * wrong_action_out["action_logits_final_train"][0, action_id].float()
                union = (output["bounded_contribution"][sample, action_id] * sign > 0).to(probability.dtype)
                union_mask = 1 - torch.prod(1 - (output["evidence_map"][sample, action_id] * union[:, None]).clamp(0, 1), dim=0)
                union_mask = union_mask * region.to(union_mask)
                keep_field = self._substitute(base_field, union_mask, region, keep_only=True)
                keep_out = model.rerun_action_evidence_from_field(
                    {"patch_tokens_by_layer_raw": keep_field}, fixed,
                    action_scale=action_scale, predicate_bias_enabled=True,
                )
                keep_margin = sign * keep_out["action_logits_final_train"][0, action_id]
                sufficiency = torch.relu(
                    self.config.sufficiency_ratio * original_margin.float() - self.config.sufficiency_margin - keep_margin.float()
                )
                selected_drops.append(selected_drop)
                control_drops.append(control_drop)
                support_values.append(signed_contribution[sample, action_id, probe_id])
                valid_values.append(torch.ones_like(selected_drop))
                sufficiency_values.append(sufficiency)
                selected_masks.append(selected_hard)
                control_masks.append(control)
                selected_logits.append(selected_out["action_logits_final_train"][0].float())
                control_logits.append(control_out["action_logits_final_train"][0].float())
                keep_masks.append(union_mask)
                keep_logits.append(keep_out["action_logits_final_train"][0].float())
                wrong_probe_masks.append((wrong_probe_mask.detach() > 0.5).to(wrong_probe_mask.dtype))
                wrong_action_masks.append((wrong_action_mask.detach() > 0.5).to(wrong_action_mask.dtype))
                wrong_probe_logits.append(wrong_probe_out["action_logits_final_train"][0].float())
                wrong_action_logits.append(wrong_action_out["action_logits_final_train"][0].float())
                wrong_probe_drops.append(wrong_probe_drop)
                wrong_action_drops.append(wrong_action_drop)
                rows.append({
                    "file_name": file_names[sample], "action_id": action_id, "probe_id": probe_id,
                    "region": region_name, "selected_control_overlap": overlap,
                    "selected_drop": float(selected_drop.detach().cpu()),
                    "control_drop": float(control_drop.detach().cpu()),
                    "selected_minus_control": float((selected_drop - control_drop).detach().cpu()),
                    "supportive_contribution": float(signed_contribution[sample, action_id, probe_id].detach().cpu()),
                    "wrong_probe_drop": float(wrong_probe_drop.detach().cpu()),
                    "wrong_action_drop": float(wrong_action_drop.detach().cpu()),
                })
                atom_count += 1
        if selected_drops:
            selected_tensor = torch.stack(selected_drops)
            control_tensor = torch.stack(control_drops)
            support_tensor = torch.stack(support_values)
            valid_tensor = torch.stack(valid_values)
            sufficiency_tensor = torch.stack(sufficiency_values)
            selected_mask_tensor = torch.stack(selected_masks)
            control_mask_tensor = torch.stack(control_masks)
            selected_logit_tensor = torch.stack(selected_logits)
            control_logit_tensor = torch.stack(control_logits)
            wrong_probe_tensor = torch.stack(wrong_probe_drops)
            wrong_action_tensor = torch.stack(wrong_action_drops)
            keep_mask_tensor = torch.stack(keep_masks)
            keep_logit_tensor = torch.stack(keep_logits)
            wrong_probe_mask_tensor = torch.stack(wrong_probe_masks)
            wrong_action_mask_tensor = torch.stack(wrong_action_masks)
            wrong_probe_logit_tensor = torch.stack(wrong_probe_logits)
            wrong_action_logit_tensor = torch.stack(wrong_action_logits)
        else:
            zero = output["action_logits_final_train"].sum() * 0
            selected_tensor = control_tensor = support_tensor = valid_tensor = sufficiency_tensor = zero.reshape(1)
            valid_tensor = torch.zeros_like(valid_tensor)
            selected_mask_tensor = control_mask_tensor = output["evidence_map"].new_zeros((0, output["evidence_map"].shape[-1]))
            selected_logit_tensor = control_logit_tensor = output["action_logits_final_train"].new_zeros((0, output["action_logits_final_train"].shape[-1]))
            wrong_probe_tensor = wrong_action_tensor = zero.reshape(1)
            keep_mask_tensor = wrong_probe_mask_tensor = wrong_action_mask_tensor = selected_mask_tensor
            keep_logit_tensor = wrong_probe_logit_tensor = wrong_action_logit_tensor = selected_logit_tensor
        return {
            "cf_valid_count": int(valid_tensor.sum().detach().cpu()),
            "cf_invalid_reason_counts": invalid,
            "selected_drop": selected_tensor,
            "control_drop": control_tensor,
            "selected_minus_control": selected_tensor - control_tensor,
            "supportive_contribution": support_tensor,
            "valid_mask": valid_tensor,
            "sufficiency_loss_raw": sufficiency_tensor,
            "selected_control_overlap": [row["selected_control_overlap"] for row in rows],
            "selected_masks": selected_mask_tensor,
            "control_masks": control_mask_tensor,
            "selected_logits": selected_logit_tensor,
            "control_logits": control_logit_tensor,
            "wrong_probe_drop": wrong_probe_tensor,
            "wrong_action_drop": wrong_action_tensor,
            "union_keep_masks": keep_mask_tensor,
            "union_keep_logits": keep_logit_tensor,
            "wrong_probe_masks": wrong_probe_mask_tensor,
            "wrong_action_masks": wrong_action_mask_tensor,
            "wrong_probe_logits": wrong_probe_logit_tensor,
            "wrong_action_logits": wrong_action_logit_tensor,
            "cases": rows,
            "dino_calls_cf_event": 0,
        }
