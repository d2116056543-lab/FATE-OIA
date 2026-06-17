from __future__ import annotations

import torch
from torch import nn


class ACPRPredicateTransportAlignment(nn.Module):
    def forward(self, predicate_attention, predicate_patch_targets):
        mass = (predicate_attention * predicate_patch_targets.float()).sum(-1)
        return {"predicate_attention_mass_on_target": mass}
