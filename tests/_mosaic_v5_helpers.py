from __future__ import annotations

import torch


def tiny_ontology(*, factor_count: int = 3) -> dict:
    factors = [{"name": f"factor_{index}"} for index in range(factor_count)]
    action_names = ("forward", "stop", "left", "right")
    action_routes = {
        action: {
            "support": [{"factor": "factor_0", "polarity": "present"}],
            "veto": [{"factor": "factor_1", "polarity": "present"}],
        }
        for action in action_names
    }
    return {
        "factors": factors,
        "factor_index": {item["name"]: index for index, item in enumerate(factors)},
        "action_names": action_names,
        "action_index": {name: index for index, name in enumerate(action_names)},
        "action_routes": action_routes,
        "reason_routes": {
            index: {"latent_factors": ["factor_0"], "absence_factors": []}
            for index in range(21)
        },
    }


def typed_inputs(*, batch_size: int = 2, factor_count: int = 3, dim: int = 8):
    coordinates = torch.zeros(batch_size, factor_count, 1, 1, 1, 2)
    features = torch.randn(batch_size, factor_count, 1, 1, 1, dim)
    attention = torch.ones(batch_size, factor_count, 1, 1, 1)
    feature_map = torch.randn(batch_size, dim, 45, 80)
    queries = torch.randn(batch_size, 4, dim)
    return feature_map, queries, coordinates, features, attention
