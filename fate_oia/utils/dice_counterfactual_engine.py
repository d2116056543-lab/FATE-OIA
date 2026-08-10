from __future__ import annotations

import torch
from torch import Tensor

from fate_oia.utils.aie_counterfactual import matched_control_mask
from .dice_counterfactual import choose_round_robin_atoms, directional_certificate, hard_region_topk


def counterfactual_logit_drop(original_delta: Tensor, modified_delta: Tensor, target_sign: Tensor) -> Tensor:
    """Measure only the intervention-induced delta change; the frozen base cancels exactly."""
    return target_sign * (original_delta - modified_delta)


class DICECounterfactualEngine:
    def __init__(self, temperature: float = .05, max_actions_per_sample: int = 2,
                 batch_fraction: float = .50, topk_patches: int = 64) -> None:
        self.temperature, self.max_actions_per_sample = float(temperature), int(max_actions_per_sample)
        self.batch_fraction, self.topk_patches = float(batch_fraction), int(topk_patches)

    @staticmethod
    def _replace(field: Tensor, mask: Tensor, background: Tensor) -> Tensor:
        return field * (1 - mask[None, None, :, None]) + background[None, None, None, :] * mask[None, None, :, None]

    def run(self, model, output: dict, action_target: Tensor, update: int) -> dict:
        selected_samples=max(1,int(output["atom_correction"].shape[0]*self.batch_fraction))
        events = choose_round_robin_atoms(output["atom_correction"], update, self.max_actions_per_sample,selected_samples)
        selected_drop, controls, atom_index = [], [], []
        for sample, action, probe in events:
            signed = 2 * action_target[sample, action] - 1
            original_delta = output["dice_action_delta"][sample, action]
            field = output["conditioned_patch_layers"][sample:sample+1].detach()
            probability = output["coherent_map"][sample, action, probe].detach()
            region_id = int(output["background_region_id"][sample, action, probe])
            region_name = model.atom_reconstructor.region_names[region_id]
            region = output["ego_region_masks"][region_name]
            selected = hard_region_topk(probability,region,self.topk_patches)
            wrong_probe = (probe + 1) % output["coherent_map"].shape[2]
            wrong_action = (action + 1) % 4
            control_1,valid_1,_=matched_control_mask(selected,region,17 + 1009*update + 97*sample + 13*action + probe,.20)
            control_2,valid_2,_=matched_control_mask(selected,region,41 + 1013*update + 101*sample + 17*action + probe,.20)
            if not (valid_1 and valid_2):
                continue
            masks = [
                (selected, region_id),
                (control_1, region_id),
                (control_2, region_id),
                (hard_region_topk(output["coherent_map"][sample, action, wrong_probe].detach(),
                                  output["ego_region_masks"][model.atom_reconstructor.region_names[int(output["background_region_id"][sample, action, wrong_probe])]],self.topk_patches),
                 int(output["background_region_id"][sample, action, wrong_probe])),
                (hard_region_topk(output["coherent_map"][sample, wrong_action, probe].detach(),
                                  output["ego_region_masks"][model.atom_reconstructor.region_names[int(output["background_region_id"][sample, wrong_action, probe])]],self.topk_patches),
                 int(output["background_region_id"][sample, wrong_action, probe])),
            ]
            drops = []
            for mask, own_region_id in masks:
                own_region = output["ego_region_masks"][model.atom_reconstructor.region_names[own_region_id]].to(field)
                bg_weight = own_region * (1 - mask)
                bg_weight = bg_weight / bg_weight.sum().clamp_min(1e-8)
                background = torch.einsum("n,lnd->d", bg_weight, field[0]).detach() / field.shape[1]
                rerun = model.rerun_dice_from_conditioned(output, self._replace(field, mask, background))
                modified_delta = rerun["dice_action_delta"][0, action]
                drops.append(counterfactual_logit_drop(original_delta, modified_delta, signed))
            selected_drop.append(drops[0]); controls.append(torch.stack(drops[1:])); atom_index.append((sample, action, probe))
        if not selected_drop:
            zero = output["action_logits_final"].sum() * 0
            return {"available": False, "selected_drop": zero[None], "controls": zero.new_zeros(1,4),
                    "license_support_cf": zero[None], "license_counter_cf": zero[None],
                    "directional_effect": zero[None], "atom_index": [], "dino_calls_cf_event": 0}
        selected_tensor, control_tensor = torch.stack(selected_drop), torch.stack(controls)
        certificate = directional_certificate(selected_tensor, *control_tensor.unbind(-1), self.temperature)
        return {"available": True, "selected_drop": selected_tensor, "controls": control_tensor,
                **certificate, "atom_index": atom_index, "dino_calls_cf_event": 0}
